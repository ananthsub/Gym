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
        --config configs/single_agent_base.yaml \
        --input-jsonl data/single_agent_10k.jsonl \
        --git-sha "$(git rev-parse --short HEAD)"

To run multiple configs back-to-back::

    python sweep_runner.py --config configs/single_agent_base.yaml configs/axis_c_8k_4mb.yaml ...
"""

from __future__ import annotations

import argparse
import os
import re
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

# How often to print a progress heartbeat while waiting for ng_run spinup.
PROGRESS_INTERVAL_S = 15.0

# Lines in ng_run.log that we surface as progress milestones. These are the patterns
# ng_run / uv emit during the long venv-create + server-spawn phase. Each tuple is
# (regex, human-readable label) — first match wins per heartbeat.
PROGRESS_PATTERNS = [
    (re.compile(r"Building venv for ([\w.-]+)"),         "building venv: {0}"),
    (re.compile(r"Resolving dependencies"),               "uv: resolving dependencies"),
    (re.compile(r"Installing\s+(\d+)\s+package"),         "uv: installing {0} packages"),
    (re.compile(r"Started server (\S+) on (http\S+)"),    "server up: {0} -> {1}"),
    (re.compile(r"(\d+)\s*/\s*(\d+)\s+servers ready"),    "servers ready: {0}/{1}"),
    (re.compile(r"All\s+\d+\s*/\s*\d+\s+servers ready"),  "ALL servers ready"),
]


def _last_progress_line(log_path: Path) -> Optional[str]:
    """Return a short human-readable summary of the most recent progress milestone in log_path.

    Returns None if the log doesn't exist yet or has no recognizable milestone.
    Cheap to call repeatedly — reads the whole file (logs are small during spinup).
    """
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None
    last_match: Optional[str] = None
    # Scan in order so the most recent matching pattern wins.
    for line in text.splitlines():
        for pat, label in PROGRESS_PATTERNS:
            m = pat.search(line)
            if m:
                last_match = label.format(*m.groups())
                break
    return last_match


def _wait_for_port(
    host: str,
    port: int,
    timeout_s: float,
    ng_proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
    phase_label: str = "head port",
) -> Tuple[bool, str]:
    """Wait for a TCP port to become connectable.

    Returns (success, reason). reason is one of:
        - "ok"                 — port bound, server reachable
        - "process_died"       — ng_proc exited before port bound
        - "timeout"            — deadline hit, port still not bound

    Prints a heartbeat every PROGRESS_INTERVAL_S with elapsed time and (if
    log_path is given) the most recent progress milestone scraped from ng_run.log.

    0.0.0.0 is a bind-all wildcard, not a normal connect target. On Linux the
    kernel translates it to 127.0.0.1 during connect(), but the behavior is
    portable-flaky — use 127.0.0.1 explicitly when the user passed 0.0.0.0.
    """
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    start = time.time()
    deadline = start + timeout_s
    next_heartbeat = start + PROGRESS_INTERVAL_S
    while time.time() < deadline:
        if ng_proc is not None and ng_proc.poll() is not None:
            return False, f"process_died (exit_code={ng_proc.returncode})"
        try:
            with socket.create_connection((connect_host, port), timeout=1.0):
                elapsed = time.time() - start
                print(f"[sweep] {phase_label}: bound after {elapsed:.1f}s", flush=True)
                return True, "ok"
        except OSError:
            pass
        now = time.time()
        if now >= next_heartbeat:
            elapsed = now - start
            milestone = _last_progress_line(log_path) if log_path is not None else None
            extra = f" — {milestone}" if milestone else ""
            print(f"[sweep] {phase_label}: waiting ({elapsed:.0f}s elapsed){extra}", flush=True)
            next_heartbeat = now + PROGRESS_INTERVAL_S
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


_RAY_GCS_PORT = 6379
_RAY_CLIENT_SERVER_PORT = 6380
_RAY_DASHBOARD_PORT = 8265
_RAY_MIN_WORKER_PORT = 2000
_RAY_MAX_WORKER_PORT = 2999
_RAY_NODE_MANAGER_PORT = 6701
_RAY_OBJECT_MANAGER_PORT = 6702
_RAY_RUNTIME_ENV_AGENT_PORT = 6703
_RAY_DASHBOARD_AGENT_GRPC_PORT = 6704
_RAY_DASHBOARD_AGENT_LISTEN_PORT = 6705
_RAY_METRICS_EXPORT_PORT = 6706


def _pre_cell_cleanup() -> None:
    """Clear leftover Ray + sub-server state on the host before launching a new cell.

    `ng_run`'s SIGINT handler tries to shut Ray down cleanly but in practice it
    leaves orphaned `raylet` / `gcs_server` / `plasma_store` / sub-server uvicorn
    processes plus stale `/tmp/ray/` session dirs. The next cell's `ng_run` then
    sees a half-alive cluster, joins it, and its sub-servers fail GCS lookup.

    Running this at the START of every cell (not at the end of the previous one)
    is idempotent across whatever exit mode the previous cell took — graceful,
    SIGKILL'd, OOM'd, manual Ctrl-C — they all look the same to the next cell.

    Cleanup steps, in order:

    1. ``pkill -9 -f`` over a broad regex covering raylet/gcs/plasma binaries,
       the `nemo_gym` / `ng_run` parent processes, the bash wrappers spawned by
       ng_run (whose cmdline carries the cd-path containing the server-type
       string), and the ray python helpers. Importantly we also include
       ``app\.py`` and ``app\.so`` so that an *orphaned* sub-server python
       interpreter is killed too: ng_run launches each sub-server as a bash
       chain that ends in ``... && python app.py``, and the python child's
       cmdline is just ``python -u app.py`` (no "synthetic_resources" path
       fragment). Without an `app.py` clause, killing bash leaves python alive
       still bound to its listening port — the next cell's ng_run then hits
       EADDRINUSE on every sub-server. Confirmed deterministically in the
       multi-agent sweep N=4 sub-sweep B cells.
    2. Port-range backstop: kill anything still listening on 5000-5999 via
       ``fuser -k`` per port. Catches anything the pkill regex missed.
    3. ``ray stop --force`` to drain the ray cluster state machine. Can take
       30-60 s after a high-N cell, so we give it 60 s and treat a timeout as
       a warning rather than aborting the cell.
    4. ``rm -rf /tmp/ray*`` to remove the stale session-dir + cluster-id
       pointers that would otherwise wedge the next ``ray.init()``.
    """
    _shell("pre-cell pkill",
           "pkill -9 -f 'raylet|gcs_server|plasma_store|ng_run|nemo_gym"
           "|synthetic_resources|synthetic_model|simple_agent"
           "|ray::|ray/_private/log_monitor|ray/autoscaler/_private/monitor"
           "|ray/dashboard/dashboard|ray.util.client.server"
           "|app\\.py|app\\.so' || true",
           timeout=30)
    # Wait briefly for the kernel to actually reap the killed processes and
    # release their listening ports before we wipe the state dirs.
    time.sleep(0.5)
    _shell("port-range fuser kill",
           # Iterate per port: `fuser -k -KILL <port>/tcp` doesn't accept ranges.
           # `command -v fuser >/dev/null` keeps this a no-op on hosts without
           # psmisc installed (rare but possible on minimal containers).
           "command -v fuser >/dev/null && "
           "for p in $(seq 5000 5999); do "
           "  fuser -k -KILL ${p}/tcp 2>/dev/null; "
           "done; true",
           timeout=30)
    _shell("ray stop --force",
           "ray stop --force 2>/dev/null || true",
           timeout=60)
    _shell("rm /tmp/ray*",
           "rm -rf /tmp/ray /tmp/ray_temp /tmp/ray-* 2>/dev/null || true",
           timeout=30)


def _shell(label: str, cmd: str, timeout: float) -> None:
    """Run a shell command for cleanup with a timeout that we treat as a warning.

    A hung cleanup step shouldn't take down the whole cell — pkill -9 may already
    have done the useful work, and the next cell's pre-cleanup will try again
    anyway. Print to stderr so the warning shows up alongside other sweep output.
    """
    try:
        subprocess.run(cmd, shell=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"[sweep] WARN: pre-cell cleanup step '{label}' timed out after "
            f"{timeout}s — continuing.",
            file=sys.stderr,
            flush=True,
        )


def _pre_start_ray() -> None:
    """Start a Ray cluster on the local host with all ports pinned below the
    kernel's ephemeral floor.

    Why pin: by default `ray.init()` (called inside `nemo_gym.server_utils.initialize_ray`)
    auto-assigns the GCS port to a random number in the 10K–40K range. On most
    Linux configurations this sits inside `net.ipv4.ip_local_port_range`
    (32768–60999 default; 9000–65000 on OCI-HSG-style hosts). With N≥128
    sub-servers each issuing parallel TCP connect()s to the GCS, the kernel
    can hand the GCS's destination port out as an ephemeral source port for an
    unrelated outgoing connection (TOCTOU between Ray's `CheckPortFree` and
    `bind`), causing `EADDRINUSE` failures.

    `ray.init()` with no args inside `ng_run` will detect the cluster started
    here (via /tmp/ray/ pointers or `RAY_ADDRESS` env if we set it) and connect
    rather than start a new one. So no `nemo_gym` change is required.

    Port layout (all below the OCI-HSG ephemeral floor of 9000):
        6379         GCS server
        6380         Ray Client server
        2000-2999    Ray worker gRPC
        6701-6706    Ray management ports (node-mgr, object-mgr, etc.)
        8265         Dashboard

    Note: gym sub-servers use 5000-5999 separately (`head_server.port` etc.);
    these don't overlap.
    """
    cmd = (
        f"ray start --head "
        f"--disable-usage-stats "
        f"--node-ip-address=127.0.0.1 "
        f"--port={_RAY_GCS_PORT} "
        f"--ray-client-server-port={_RAY_CLIENT_SERVER_PORT} "
        f"--dashboard-port={_RAY_DASHBOARD_PORT} "
        f"--include-dashboard=False "
        f"--min-worker-port={_RAY_MIN_WORKER_PORT} "
        f"--max-worker-port={_RAY_MAX_WORKER_PORT} "
        f"--node-manager-port={_RAY_NODE_MANAGER_PORT} "
        f"--object-manager-port={_RAY_OBJECT_MANAGER_PORT} "
        f"--runtime-env-agent-port={_RAY_RUNTIME_ENV_AGENT_PORT} "
        f"--dashboard-agent-grpc-port={_RAY_DASHBOARD_AGENT_GRPC_PORT} "
        f"--dashboard-agent-listen-port={_RAY_DASHBOARD_AGENT_LISTEN_PORT} "
        f"--metrics-export-port={_RAY_METRICS_EXPORT_PORT}"
    )
    print(f"[sweep] pre-starting Ray with pinned ports: GCS={_RAY_GCS_PORT}", flush=True)
    rc = subprocess.run(cmd, shell=True, check=False, timeout=120).returncode
    if rc != 0:
        print(
            f"[sweep] WARNING: pre-started ray exited rc={rc}. "
            f"`ng_run` will fall back to auto-assigning GCS port (may collide with ephemeral range at high N).",
            flush=True,
        )


def _run_one_cell(
    config_path: Path,
    input_jsonl: Optional[Path],
    output_dir: Path,
    head_server_host: str,
    head_server_port: int,
    spinup_timeout_s: float,
    driver_mode: str = "loaded",
    idle_window_s: float = 30.0,
    teardown_sleep_s: float = 5.0,
) -> int:
    """Returns 0 on success, non-zero on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Wipe inherited Ray + sub-server state before launch. See _pre_cell_cleanup
    # docstring — this is what makes back-to-back cells reliable, replacing the
    # earlier "trust the previous cell tore itself down" approach.
    print("[sweep] pre-cell cleanup (pkill ray/gym + rm /tmp/ray)", flush=True)
    _pre_cell_cleanup()

    # 0.5. Pre-start Ray with pinned ports so ng_run's `ray.init()` connects to
    # an existing cluster on known-safe ports instead of auto-assigning into the
    # kernel ephemeral range. Critical at high N (~128+) where TOCTOU port
    # collisions cause sub-server `ray.init()` failures — see investigations/
    # nemo-gym-scale-testing.md §10.3.4 (A13). At low N this is a no-op
    # quality-of-life improvement.
    _pre_start_ray()

    ng_log = (output_dir / "ng_run.log").open("w")
    driver_log = (output_dir / "driver.log").open("w")

    # Save the resolved config alongside results for reproducibility.
    (output_dir / "config.yaml").write_text(config_path.read_text())

    # 1. ng_run.
    # PYTHONUNBUFFERED=1 makes the child's stdout line-buffered so milestones
    # land in ng_run.log promptly (otherwise they sit in the child's 4 KB block
    # buffer for tens of seconds and the heartbeat scraper can't see them).
    ng_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    ng_cmd = f'ng_run "+config_paths=[{config_path.as_posix()}]"'
    print(f"[sweep] launching: {ng_cmd} (cwd={SCALE_SIM_DIR})", flush=True)
    ng_proc = subprocess.Popen(
        ng_cmd,
        shell=True,
        cwd=SCALE_SIM_DIR,
        stdout=ng_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=ng_env,
    )

    ng_log_path = output_dir / "ng_run.log"
    print(
        f"[sweep] cell={config_path.name} pid={ng_proc.pid} ng_run.log={ng_log_path}",
        flush=True,
    )
    print(
        f"[sweep] tip: in another shell, `tail -f {ng_log_path}` for full sub-server output.",
        flush=True,
    )
    try:
        ok, reason = _wait_for_port(
            head_server_host,
            head_server_port,
            timeout_s=spinup_timeout_s,
            ng_proc=ng_proc,
            log_path=ng_log_path,
            phase_label=f"head:{head_server_port}",
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

        # 2. load driver — stream its stdout to BOTH the log file and the foreground
        # so the periodic window-summary prints (every 10s) are visible while the
        # cell runs, not just after it finishes.
        driver_log_path = output_dir / "driver.log"
        driver_parts = [
            shlex.quote(sys.executable),
            "-u",
            shlex.quote(str(SCALE_SIM_DIR / "load_driver.py")),
            "--config", shlex.quote(str(config_path)),
            "--output-dir", shlex.quote(str(output_dir)),
            "--head-server-host", shlex.quote(head_server_host),
            "--head-server-port", str(head_server_port),
            "--mode", driver_mode,
        ]
        if driver_mode == "spinup_only":
            driver_parts += ["--idle-window-s", str(idle_window_s)]
        else:
            if input_jsonl is None:
                raise ValueError("driver_mode='loaded' requires input_jsonl to be set.")
            driver_parts += ["--input-jsonl", shlex.quote(str(input_jsonl))]
        driver_cmd = " ".join(driver_parts)
        print(f"[sweep] launching: {driver_cmd}", flush=True)
        driver_log.close()  # reopen unbuffered for the streaming loop below
        with driver_log_path.open("w", buffering=1) as dlog:
            driver_proc = subprocess.Popen(
                driver_cmd,
                shell=True,
                cwd=SCALE_SIM_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert driver_proc.stdout is not None
            for line in driver_proc.stdout:
                dlog.write(line)
                sys.stdout.write(f"[driver] {line}")
                sys.stdout.flush()
            driver_proc.wait()
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
        # driver_log is closed inside the streaming loop above (or never opened if we aborted early).
        try:
            driver_log.close()
        except Exception:
            pass
        # Brief pause so kernel TIME_WAIT entries from this cell drain. Was load-bearing
        # before _pre_cell_cleanup() existed (we relied on graceful shutdown to reap
        # state); now it's just defensive against TCP TIME_WAIT pressure on the next
        # cell's port allocations. Keep it short — _pre_cell_cleanup() handles the
        # state-clearing job, and TIME_WAIT entries on closed ports don't block new
        # binds on the same ports (they only consume kernel TCP table slots).
        time.sleep(teardown_sleep_s)


def _wait_for_servers_ready(
    log_path: Path,
    timeout_s: float,
    ng_proc: Optional[subprocess.Popen] = None,
) -> bool:
    """Poll ng_run.log until 'All N / N servers ready!' appears.

    Prints a heartbeat every PROGRESS_INTERVAL_S with elapsed time and the most
    recent milestone (which sub-server is currently building venv / spinning up).

    Exits early if ng_proc dies — there's no point waiting for a ready signal
    from a dead process.
    """
    start = time.time()
    deadline = start + timeout_s
    next_heartbeat = start + PROGRESS_INTERVAL_S
    last_milestone: Optional[str] = None
    while time.time() < deadline:
        if ng_proc is not None and ng_proc.poll() is not None:
            return False
        if log_path.exists():
            try:
                tail = log_path.read_text(errors="replace")
                if "servers ready! Polling every 60s" in tail:
                    elapsed = time.time() - start
                    print(f"[sweep] sub-servers ready after {elapsed:.1f}s", flush=True)
                    return True
            except Exception:
                pass
        now = time.time()
        if now >= next_heartbeat:
            elapsed = now - start
            milestone = _last_progress_line(log_path)
            # Print every heartbeat so the user always knows the cell is alive,
            # even if the milestone hasn't advanced.
            tag = milestone or "(no milestone yet — likely uv resolve / install)"
            changed = " [new]" if milestone and milestone != last_milestone else ""
            print(f"[sweep] sub-servers: waiting ({elapsed:.0f}s elapsed) — {tag}{changed}", flush=True)
            last_milestone = milestone
            next_heartbeat = now + PROGRESS_INTERVAL_S
        time.sleep(2.0)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="One or more sweep config YAMLs.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=False,
        default=None,
        help="JSONL of input rows. Reused across all cells. Required for --driver-mode=loaded; ignored for --driver-mode=spinup_only.",
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
            "so the timeout only matters for genuine slow startup (cold uv sync on Lustre, large N)."
        ),
    )
    parser.add_argument(
        "--driver-mode",
        choices=("loaded", "spinup_only"),
        default="loaded",
        help="loaded (default): drive concurrent /run requests. spinup_only: idle sample only (multi-agent pre-flight).",
    )
    parser.add_argument(
        "--idle-window-s",
        type=float,
        default=30.0,
        help="With --driver-mode=spinup_only, how long the driver samples after sub-servers are ready.",
    )
    parser.add_argument(
        "--teardown-sleep-s",
        type=float,
        default=5.0,
        help="Sleep between cells to let kernel TIME_WAIT drain. Bump to ~N//8 for high-N multi-agent cells.",
    )
    args = parser.parse_args()
    if args.driver_mode == "loaded" and args.input_jsonl is None:
        parser.error("--input-jsonl is required when --driver-mode=loaded.")

    results_root = SCALE_SIM_DIR / "results" / args.git_sha
    results_root.mkdir(parents=True, exist_ok=True)

    failures: List[Path] = []
    for cfg_path in args.config:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = results_root / f"{cfg_path.stem}_{ts}"
        rc = _run_one_cell(
            config_path=cfg_path.resolve(),
            input_jsonl=args.input_jsonl.resolve() if args.input_jsonl else None,
            output_dir=output_dir,
            head_server_host=args.head_server_host,
            head_server_port=args.head_server_port,
            spinup_timeout_s=args.spinup_timeout_s,
            driver_mode=args.driver_mode,
            idle_window_s=args.idle_window_s,
            teardown_sleep_s=args.teardown_sleep_s,
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
