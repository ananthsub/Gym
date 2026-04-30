# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep runner — orchestrates ng_run + load_driver across one or more configs.

Each cell of the sweep is one ``ng_run`` invocation followed by one
``load_driver.py`` invocation, with full teardown between cells. Process state
(FD pool, kernel TCP TIME_WAIT, aiohttp connector) does NOT survive between
cells, on purpose — we want each cell to start clean.

Usage::

    cd tools/scale_sim/
    python sweep_runner.py \
        --config configs/axis_a_8k.yaml \
        --input-jsonl data/axis_a_10k.jsonl \
        --git-sha "$(git rev-parse --short HEAD)"

To run multiple configs back-to-back::

    python sweep_runner.py --config configs/axis_a_8k.yaml configs/axis_c_8k_4mb.yaml ...
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SCALE_SIM_DIR = Path(__file__).resolve().parent


def _wait_for_port(
    host: str,
    port: int,
    timeout_s: float,
    ng_proc: Optional[subprocess.Popen] = None,
) -> Tuple[bool, str]:
    """Wait for a TCP port to become connectable.

    Returns (success, reason). reason is one of:
        - "ok"                 — port bound, server reachable
        - "process_died"       — ng_proc exited before port bound
        - "timeout"            — deadline hit, port still not bound

    0.0.0.0 is a bind-all wildcard, not a normal connect target. On Linux the
    kernel translates it to 127.0.0.1 during connect(), but the behavior is
    portable-flaky — use 127.0.0.1 explicitly when the user passed 0.0.0.0.
    """
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ng_proc is not None and ng_proc.poll() is not None:
            return False, f"process_died (exit_code={ng_proc.returncode})"
        try:
            with socket.create_connection((connect_host, port), timeout=1.0):
                return True, "ok"
        except OSError:
            time.sleep(1.0)
    return False, "timeout"


def _tail_log(path: Path, n_lines: int = 60) -> str:
    """Return the last n_lines of a log file. Best-effort; returns "" on errors."""
    if not path.exists():
        return f"(log file does not exist: {path})"
    try:
        # Read the whole file — the logs are bounded by ng_run runtime so this is fine.
        text = path.read_text(errors="replace")
    except Exception as e:
        return f"(failed to read {path}: {e})"
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def _run_one_cell(
    config_path: Path,
    input_jsonl: Path,
    output_dir: Path,
    head_server_host: str,
    head_server_port: int,
    spinup_timeout_s: float,
) -> int:
    """Returns 0 on success, non-zero on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ng_log = (output_dir / "ng_run.log").open("w")
    driver_log = (output_dir / "driver.log").open("w")

    # Save the resolved config alongside results for reproducibility.
    (output_dir / "config.yaml").write_text(config_path.read_text())

    # 1. ng_run
    ng_cmd = f'ng_run "+config_paths=[{config_path.as_posix()}]"'
    print(f"[sweep] launching: {ng_cmd} (cwd={SCALE_SIM_DIR})", flush=True)
    ng_proc = subprocess.Popen(
        ng_cmd,
        shell=True,
        cwd=SCALE_SIM_DIR,
        stdout=ng_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    ng_log_path = output_dir / "ng_run.log"
    try:
        ok, reason = _wait_for_port(
            head_server_host,
            head_server_port,
            timeout_s=spinup_timeout_s,
            ng_proc=ng_proc,
        )
        if not ok:
            print(
                f"[sweep] head server did not come up on {head_server_host}:{head_server_port} "
                f"({reason}). Aborting cell.\n"
                f"--- last 60 lines of {ng_log_path} ---\n"
                f"{_tail_log(ng_log_path)}\n"
                f"--- end of log tail ---",
                flush=True,
            )
            return 1

        # Give the sub-servers time to spin up after the head is reachable.
        # ng_run prints "All N / N servers ready!" on stdout when done; tail the log.
        if not _wait_for_servers_ready(ng_log_path, timeout_s=spinup_timeout_s, ng_proc=ng_proc):
            print(
                "[sweep] sub-servers did not signal ready in time. Aborting cell.\n"
                f"--- last 60 lines of {ng_log_path} ---\n"
                f"{_tail_log(ng_log_path)}\n"
                f"--- end of log tail ---",
                flush=True,
            )
            return 1

        # 2. load driver
        driver_cmd = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(SCALE_SIM_DIR / 'load_driver.py'))} "
            f"--config {shlex.quote(str(config_path))} "
            f"--input-jsonl {shlex.quote(str(input_jsonl))} "
            f"--output-dir {shlex.quote(str(output_dir))} "
            f"--head-server-host {shlex.quote(head_server_host)} "
            f"--head-server-port {head_server_port}"
        )
        print(f"[sweep] launching: {driver_cmd}", flush=True)
        driver_proc = subprocess.run(
            driver_cmd,
            shell=True,
            cwd=SCALE_SIM_DIR,
            stdout=driver_log,
            stderr=subprocess.STDOUT,
        )
        return driver_proc.returncode

    finally:
        # 3. teardown
        print("[sweep] tearing down ng_run process group", flush=True)
        try:
            os.killpg(os.getpgid(ng_proc.pid), signal.SIGINT)
            ng_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(ng_proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        ng_log.close()
        driver_log.close()
        # Brief pause so kernel TIME_WAIT entries from this cell don't bleed into the next.
        time.sleep(5.0)


def _wait_for_servers_ready(
    log_path: Path,
    timeout_s: float,
    ng_proc: Optional[subprocess.Popen] = None,
) -> bool:
    """Poll ng_run.log until 'All N / N servers ready!' appears.

    Exits early if ng_proc dies — there's no point waiting for a ready signal
    from a dead process.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ng_proc is not None and ng_proc.poll() is not None:
            return False
        if log_path.exists():
            try:
                tail = log_path.read_text(errors="replace")
                if "servers ready! Polling every 60s" in tail:
                    return True
            except Exception:
                pass
        time.sleep(2.0)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="One or more sweep config YAMLs.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="JSONL of input rows. Reused across all cells.",
    )
    parser.add_argument("--git-sha", type=str, default="local", help="Tag for results directory.")
    parser.add_argument("--head-server-host", default="0.0.0.0")
    parser.add_argument("--head-server-port", type=int, default=5000)
    parser.add_argument(
        "--spinup-timeout-s",
        type=float,
        default=300.0,
        help=(
            "Per-cell spinup timeout. We exit early if ng_run dies before the port binds, "
            "so the timeout only matters for genuine slow startup (cold uv sync on Lustre)."
        ),
    )
    args = parser.parse_args()

    results_root = SCALE_SIM_DIR / "results" / args.git_sha
    results_root.mkdir(parents=True, exist_ok=True)

    failures: List[Path] = []
    for cfg_path in args.config:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = results_root / f"{cfg_path.stem}_{ts}"
        rc = _run_one_cell(
            config_path=cfg_path.resolve(),
            input_jsonl=args.input_jsonl.resolve(),
            output_dir=output_dir,
            head_server_host=args.head_server_host,
            head_server_port=args.head_server_port,
            spinup_timeout_s=args.spinup_timeout_s,
        )
        print(f"[sweep] cell {cfg_path.name} → rc={rc} → {output_dir}", flush=True)
        if rc != 0:
            failures.append(cfg_path)

    if failures:
        print(f"\n[sweep] {len(failures)} / {len(args.config)} cells failed: {[str(p) for p in failures]}")
        sys.exit(1)
    else:
        print(f"\n[sweep] all {len(args.config)} cells succeeded. Results under {results_root}")


if __name__ == "__main__":
    main()
