# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ray-actor wrapper around a gym sub-server topology.

The training framework wraps NeMo Gym in a Ray actor: the trainer calls the
actor with blocking ``ray.get``, and the actor fans rollouts out to the gym
servers over HTTP. The direct load driver (``load_driver.py``) talks HTTP to
the head server and skips the Ray boundary, so it cannot reproduce failures
that depend on that boundary. This actor adds it back.

What the actor does:

- On ``__init__``, spawns ``ng_run`` as a child process group bringing up a
  single-agent topology (head + model + resources + agent) from a YAML config.
  ``wait_ready`` blocks until the servers report ready.
- ``collect_batch(rows)``: accept a batch of rollout rows, fan them out
  concurrently with ``asyncio`` against the head server, and return the whole
  batch result as one object. This is the shape the trainer's blocking
  ``ray.get`` wraps, and the return object crosses Ray's object store.
- ``collect_batch_streaming(rows)``: same fan-out, but yields each row as it
  completes so the trainer can consume results incrementally.

What the actor does not do:

- It does not replace the agent. The wrapped topology runs the full gym stack
  end to end (head -> agent -> model + resources).
- It does not skip serialization. Each result crosses Ray's object-store
  transport, which is the path under test.

Usage from the mock trainer:

    import ray
    from gym_actor import NemoGymActor

    ray.init(address="auto")
    actor = NemoGymActor.options(name="gym").remote(
        config_path="configs/actor_repro.yaml",
        head_server_host="127.0.0.1",
        head_server_port=5000,
    )
    ray.get(actor.wait_ready.remote(), timeout=600)
    batch_result = ray.get(actor.collect_batch.remote(rows))
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
import ray
from aiohttp import ClientPayloadError


SCALE_SIM_DIR = Path(__file__).resolve().parent


def _wait_for_port(host: str, port: int, deadline: float) -> bool:
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    while time.time() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _wait_for_servers_ready_in_log(log_path: Path, deadline: float) -> bool:
    """Match sweep_runner._wait_for_servers_ready's "servers ready! Polling" signal."""
    while time.time() < deadline:
        if log_path.exists():
            try:
                if "servers ready! Polling every 60s" in log_path.read_text(errors="replace"):
                    return True
            except Exception:
                pass
        time.sleep(1.0)
    return False


@ray.remote
class NemoGymActor:
    """Single-agent gym wrapped as a Ray actor.

    Each actor instance owns one ``ng_run`` process group; that process group
    owns the head server + simple_agent + synthetic_model + synthetic_resources
    sub-servers brought up from ``config_path``. The actor itself does not run
    a FastAPI server — it just holds an aiohttp session that talks to the head
    server over loopback.
    """

    def __init__(
        self,
        config_path: str,
        head_server_host: str = "127.0.0.1",
        head_server_port: int = 5000,
        log_dir: Optional[str] = None,
        spinup_timeout_s: float = 600.0,
        in_actor_retry_payload_error: bool = False,
        in_actor_retry_sleep_s: float = 0.5,
    ) -> None:
        self.config_path = Path(config_path)
        self.head_server_host = head_server_host
        self.head_server_port = head_server_port
        self.spinup_timeout_s = spinup_timeout_s
        # Matches the production gym RolloutCollectionHelper._post_subroutine
        # behavior: catch ClientPayloadError, sleep, retry indefinitely. Off
        # by default so failures are visible; turn on to mirror the production
        # in-actor retry path.
        self.in_actor_retry_payload_error = in_actor_retry_payload_error
        self.in_actor_retry_sleep_s = in_actor_retry_sleep_s

        # Distinguish per-actor log dirs so concurrent actors don't fight over ng_run.log.
        if log_dir is None:
            log_dir = str(SCALE_SIM_DIR / "results" / "actor_logs" / f"actor_{os.getpid()}")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ng_log_path = self.log_dir / "ng_run.log"

        # Lazy-imported aiohttp / nemo_gym pieces — the actor module imports
        # need to stay light so ``ray.put``-side imports don't pull a webserver
        # stack into the trainer driver.
        from nemo_gym.config_types import BaseServerConfig
        from nemo_gym.server_utils import (
            GlobalAIOHTTPAsyncClientConfig,
            ServerClient,
            get_response_json,
            is_global_aiohttp_client_setup,
            raise_for_status,
            set_global_aiohttp_client,
        )

        self._ServerClient = ServerClient
        self._set_global_aiohttp_client = set_global_aiohttp_client
        self._is_global_aiohttp_client_setup = is_global_aiohttp_client_setup
        self._GlobalAIOHTTPAsyncClientConfig = GlobalAIOHTTPAsyncClientConfig
        self._BaseServerConfig = BaseServerConfig
        self._raise_for_status = raise_for_status
        self._get_response_json = get_response_json

        self.head_server_config = BaseServerConfig(host=head_server_host, port=head_server_port)
        self.server_client = None  # filled in by wait_ready()

        # Spawn ng_run synchronously. The constructor does not return until the
        # subprocess is created; readiness check waits in wait_ready().
        ng_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        ng_cmd = f'ng_run "+config_paths=[{self.config_path.as_posix()}]"'
        self._ng_log_fh = self.ng_log_path.open("w")
        self.ng_proc = subprocess.Popen(
            ng_cmd,
            shell=True,
            cwd=SCALE_SIM_DIR,
            stdout=self._ng_log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=ng_env,
        )
        self._ready = False

    def wait_ready(self) -> Dict[str, Any]:
        """Block until ng_run reports all sub-servers ready, then build the aiohttp client.

        Returns a small status dict for the trainer to log. Raises RuntimeError
        on timeout or on ng_run dying.
        """
        deadline = time.time() + self.spinup_timeout_s
        # 1. head port becomes bindable
        if not _wait_for_port(self.head_server_host, self.head_server_port, deadline):
            self._fail_actor("head_server did not bind", deadline)
        # 2. all sub-servers signal ready via the log
        if not _wait_for_servers_ready_in_log(self.ng_log_path, deadline):
            self._fail_actor("sub-servers did not signal ready", deadline)
        # 3. construct ServerClient — same path load_driver uses
        sc = self._ServerClient.load_from_global_config(self.head_server_config)
        if not self._is_global_aiohttp_client_setup():
            self._set_global_aiohttp_client(
                cfg=self._GlobalAIOHTTPAsyncClientConfig.model_validate(sc.global_config_dict)
            )
        self.server_client = sc
        self._ready = True
        elapsed = self.spinup_timeout_s - max(0.0, deadline - time.time())
        return {
            "ready": True,
            "head_server_port": self.head_server_port,
            "spinup_elapsed_s": elapsed,
            "ng_log_path": str(self.ng_log_path),
        }

    def _fail_actor(self, reason: str, deadline: float) -> None:
        # Try to surface a useful tail of the log on actor failure.
        tail = ""
        try:
            text = self.ng_log_path.read_text(errors="replace")
            tail = "\n".join(text.splitlines()[-60:])
        except Exception:
            tail = "(could not read ng_run.log)"
        elapsed = self.spinup_timeout_s - max(0.0, deadline - time.time())
        raise RuntimeError(
            f"NemoGymActor spinup failed: {reason} after {elapsed:.1f}s. Last 60 lines of {self.ng_log_path}:\n{tail}"
        )

    async def _run_one_row(self, idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
        """Single-row coroutine shared by collect_batch and collect_batch_streaming.

        Mirrors the production gym ``RolloutCollectionHelper._post_subroutine``
        when ``in_actor_retry_payload_error=True``: catches ``ClientPayloadError``
        only, sleeps, retries indefinitely. Other errors propagate immediately
        so the trainer can decide retry policy (matches the production split
        between in-actor retry of payload-framing errors and trainer-side
        whole-prompt-group retry of everything else).
        """
        agent_name = row["agent_ref"]["name"]
        t0 = time.perf_counter()
        n_payload_retries = 0
        while True:
            try:
                res = await self.server_client.post(
                    server_name=agent_name,
                    url_path="/run",
                    json=row,
                )
                await self._raise_for_status(res)
                resp_json = await self._get_response_json(res)
                response_bytes = len(orjson.dumps(resp_json)) if resp_json is not None else 0
                return {
                    "row_idx": idx,
                    "status": "ok",
                    "latency_s": time.perf_counter() - t0,
                    "response_bytes": response_bytes,
                    "n_payload_retries": n_payload_retries,
                }
            except ClientPayloadError as e:
                if not self.in_actor_retry_payload_error:
                    return {
                        "row_idx": idx,
                        "status": "error",
                        "latency_s": time.perf_counter() - t0,
                        "error_class": type(e).__name__,
                        "error_message": str(e)[:500],
                        "n_payload_retries": n_payload_retries,
                    }
                n_payload_retries += 1
                await asyncio.sleep(self.in_actor_retry_sleep_s)
                continue
            except Exception as e:
                return {
                    "row_idx": idx,
                    "status": "error",
                    "latency_s": time.perf_counter() - t0,
                    "error_class": type(e).__name__,
                    "error_message": str(e)[:500],
                    "n_payload_retries": n_payload_retries,
                }

    async def collect_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fan out N concurrent /run requests, return the whole batch as one object.

        This is the code path the trainer's blocking ``ray.get`` wraps. The
        single return object crosses Ray's object-store transport, which is the
        path where the production ``ClientPayloadError`` was observed under high
        trainer-side thread counts.

        Returns one dict per input row, in input order, with keys row_idx,
        status, latency_s, response_bytes, and (on failure) error_class and
        error_message. The actor does not retry transient errors here.
        """
        if not self._ready:
            raise RuntimeError("NemoGymActor.collect_batch called before wait_ready()")
        assert self.server_client is not None
        return await asyncio.gather(*(self._run_one_row(i, r) for i, r in enumerate(rows)))

    async def collect_batch_streaming(self, rows: List[Dict[str, Any]]):
        """Fan out N concurrent /run requests and yield each row as it completes.

        Same per-row coroutine as ``collect_batch``, but exposed as an async
        generator. Ray turns this into an ``ObjectRefGenerator`` on the trainer
        side, so each ``yield`` is one ObjectRef the trainer can ``ray.get``
        independently. This is the streaming-return shape, and it avoids two
        problems of the single-object return:

        1. The whole batch crossing the object store as one transfer (the
           ClientPayloadError path) becomes many small per-rollout transfers.
        2. The trainer can size its thread pool to the lag it tolerates
           (typically a handful) instead of to the number of prompt groups
           (hundreds to thousands), cutting Ray-RPC concurrency sharply.

        The fan-out inside the actor is unchanged, so the model still sees the
        same N concurrent in-flight requests; only the trainer/actor return
        boundary differs.
        """
        if not self._ready:
            raise RuntimeError("NemoGymActor.collect_batch_streaming called before wait_ready()")
        assert self.server_client is not None

        pending = {asyncio.create_task(self._run_one_row(i, r)): i for i, r in enumerate(rows)}
        while pending:
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pending.pop(task)
                yield await task

    def shutdown(self) -> None:
        """SIGINT the ng_run process group and wait briefly for graceful shutdown."""
        if self.ng_proc is not None and self.ng_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.ng_proc.pid), signal.SIGINT)
                self.ng_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.ng_proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        try:
            self._ng_log_fh.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
