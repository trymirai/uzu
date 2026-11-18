"""Compare uzu and ML Drift on the same Qwen3 checkpoint and request.

At 4 bits, uzu runs a lalamo conversion of mlx-community/<model>-4bit (group 64, scale and
bias; uzu's GEMM has no group 128 variant for Qwen's own MLX checkpoints) and ML Drift runs
upstream's Q4_0 extraction (group 32, scale only). At 16 bits, uzu runs a lalamo bf16
conversion and ML Drift its unquantized extraction, uploaded as fp16. Both decode greedily.
Each engine first answers one run of the request as a warmup, so build and model loading
finish before the optional cooldown and the measured runs.
"""

import json
import subprocess
from pathlib import Path
from statistics import fmean
from typing import Annotated

import typer
from bench import BenchRequest, ChatMessage
from engines import ROOT, prompt_tokens, run_engine
from server.main import get_bench_average
from server.util import load_input

UZU_MODELS = ROOT / "workspace/uzu-models"

app = typer.Typer(add_completion=False)


CONVERT = """
import dataclasses, sys
from pathlib import Path
from lalamo.commands import convert
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.model_registry import ModelRegistry
spec = ModelRegistry.build().repo_to_model[sys.argv[1]]
convert(dataclasses.replace(spec, origin=HuggingFaceOrigin(repo=sys.argv[2])), Path(sys.argv[3]))
"""


def prepare_uzu_model(model: str, bits: int) -> Path:
    name = model.split("/")[-1]
    spec, origin = (f"{model}-MLX-4bit", f"mlx-community/{name}-4bit") if bits == 4 else (model, model)
    path = UZU_MODELS / origin.replace("/", "--")
    if not (path / "config.json").is_file():
        command = ["uvx", "--python", "3.13", "--from", "lalamo@0.17.0", "python", "-c", CONVERT]
        subprocess.run([*command, spec, origin, str(path)], check=True)
    (path / "encoding.json").write_text(json.dumps([{"type": "hanashi", "name": "qwen3"}]) + "\n")
    return path


@app.command()
def main(
    model: Annotated[str, typer.Option(help="Qwen3 checkpoint with an ML Drift preset.")] = "Qwen/Qwen3-14B",
    bits: Annotated[int, typer.Option(help="Weight bits for both engines: 4 or 16.")] = 4,
    input: Annotated[Path, typer.Option(help="Task JSON with 'messages'.")] = ROOT / "server/tasks/london-summary.json",
    num_runs: Annotated[int, typer.Option(min=1)] = 5,
    max_tokens: Annotated[int, typer.Option(min=2)] = 256,
    wait_cooldown: Annotated[
        bool, typer.Option("--wait-cooldown/--no-wait-cooldown", help="Wait for CPU and GPU <= 60°C before measuring.")
    ] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for each response, including builds.")] = 3600,
) -> None:
    if bits not in (4, 16):
        raise typer.BadParameter("Use 4 or 16.", param_hint="--bits")
    uzu_model = prepare_uzu_model(model, bits)
    messages = [ChatMessage.model_validate(message) for message in load_input(input)["messages"]]
    request = BenchRequest(prompt_chat=messages, max_tokens=max_tokens, num_runs=num_runs)
    engines = {
        "uzu": ["cargo", "run", "--release", "-p", "benchmarks-uzu", "--", "--model", str(uzu_model)],
        "mldrift": ["./engine-mldrift/run.sh", "--model", model, "--precision", "q4_0" if bits == 4 else "f16"],
    }
    results = {
        name: run_engine(name, command, [request], wait_cooldown, timeout)[0] for name, command in engines.items()
    }

    header = f"{'engine':<8} {'prompt':>6} {'output':>6} {'TTFT ms':>8} {'prefill t/s':>11} {'decode t/s':>10}"
    print(f"{header} {'total s':>7} {'phys MiB':>8} {'GPU MiB':>8}")
    averages = {}
    for name, responses in results.items():
        average = averages[name] = get_bench_average(responses)
        print(
            f"{name:<8} {prompt_tokens(responses[0]):>6} {fmean(r.tokens_count for r in responses):>6.0f} "
            f"{average.time_to_first_token * 1000:>8.2f} {average.prompt_tps:>11.1f} {average.decode_tps:>10.1f} "
            f"{average.duration:>7.3f} {average.max_memory_phys_footprint / 2**20:>8.0f} "
            f"{average.max_memory_graphics_total / 2**20:>8.0f}"
        )
    uzu, mldrift = averages["uzu"], averages["mldrift"]
    print(
        f"uzu / ML Drift: prefill {uzu.prompt_tps / mldrift.prompt_tps:.2f}x, "
        f"decode {uzu.decode_tps / mldrift.decode_tps:.2f}x, "
        f"TTFT {uzu.time_to_first_token / mldrift.time_to_first_token:.2f}x"
    )

    counts = {name: {prompt_tokens(response) for response in responses} for name, responses in results.items()}
    if len(counts["uzu"] | counts["mldrift"]) != 1:
        typer.echo(f"Warning: prompt token counts differ between engines: {counts}", err=True)
    for name, responses in results.items():
        typer.echo(f"\n{name} text: {responses[0].text[:300]!r}", err=True)


if __name__ == "__main__":
    app()
