from __future__ import annotations

import subprocess
import time
from pathlib import Path


def _xctrace_cmd(*args: str) -> list[str]:
    return ["xcrun", "xctrace", *args]


def record_trace(
    *,
    output: Path,
    command: list[str],
    template: str = "Metal System Trace",
    gpu_counters: bool = False,
    time_limit: str | None = None,
    env: list[str] | None = None,
    attach_pid: int | None = None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Output trace already exists: {output}")

    cmd = _xctrace_cmd(
        "record",
        "--template",
        template,
        "--output",
        str(output),
        "--no-prompt",
        "--target-stdout",
        "-",
    )

    if gpu_counters:
        cmd += ["--instrument", "Metal GPU Counters"]

    if time_limit:
        cmd += ["--time-limit", time_limit]

    for item in env or []:
        cmd += ["--env", item]

    if attach_pid is not None:
        cmd += ["--attach", str(attach_pid)]
    elif command:
        cmd += ["--launch", "--", *command]
    else:
        raise SystemExit("Either command or --attach must be specified")

    proc = subprocess.run(cmd)
    if not output.exists():
        return proc.returncode
    return 0


def default_output_path() -> Path:
    return Path(f"gpu_trace_{int(time.time())}.trace")
