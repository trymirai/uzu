from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .comparison import TraceProfile, load_trace_profile


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_path: str


@dataclass(frozen=True)
class RunConfig:
    cli_binary: str
    gpu_trace_dir: str
    task: dict
    models: list[ModelSpec]

    @classmethod
    def from_file(cls, path: Path | str) -> RunConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(
            cli_binary=data["cli_binary"],
            gpu_trace_dir=data.get("gpu_trace_dir", ""),
            task=data["task"],
            models=[ModelSpec(**m) for m in data["models"]],
        )


def record_trace_for_model(
    spec: ModelSpec,
    cli_binary: Path,
    gpu_trace_dir: Path,
    task: dict,
    output_dir: Path,
    console: Console,
) -> Path:
    task_file = output_dir / f"{spec.label.replace(' ', '_')}_task.json"
    task_file.write_text(json.dumps(task, indent=2))

    result_file = output_dir / f"{spec.label.replace(' ', '_')}_result.json"
    trace_file = output_dir / f"{spec.label.replace(' ', '_')}.trace"
    json_file = trace_file.with_suffix(".json")

    console.print(f"[cyan]Recording trace for {spec.label}...[/cyan]")

    cmd = [
        "uv", "run", "gpu_trace", "run",
        "-v", "--gpu-counters",
        "--json", str(json_file),
        "-o", str(trace_file),
        "--",
        str(cli_binary), "bench",
        spec.model_path,
        str(task_file),
        str(result_file),
    ]

    proc = subprocess.run(cmd, cwd=str(gpu_trace_dir), capture_output=True, text=True)

    if proc.returncode != 0:
        console.print(f"[red]Error recording {spec.label}:[/red]")
        console.print(proc.stderr[-500:] if proc.stderr else "(no stderr)")
        raise RuntimeError(f"gpu_trace failed for {spec.label}")

    console.print(f"[green]Done: {spec.label} -> {json_file}[/green]")
    return json_file


def record_all(config: RunConfig, output_dir: Path | None = None, console: Console | None = None) -> list[TraceProfile]:
    console = console or Console()

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="model_compare_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_binary = Path(config.cli_binary).resolve()
    gpu_trace_dir = Path(config.gpu_trace_dir).resolve() if config.gpu_trace_dir else Path.cwd()

    profiles: list[TraceProfile] = []
    for spec in config.models:
        json_path = record_trace_for_model(
            spec=spec,
            cli_binary=cli_binary,
            gpu_trace_dir=gpu_trace_dir,
            task=config.task,
            output_dir=output_dir,
            console=console,
        )
        profile = load_trace_profile(json_path, spec.label)
        profiles.append(profile)

    return profiles
