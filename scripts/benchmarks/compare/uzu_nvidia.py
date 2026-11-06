"""Compare uzu with CUDA engines on one NVIDIA GPU on Windows, each with its own speculative decoding.

uzu runs a trymirai checkpoint with its speculator on Vulkan. Inside WSL on CUDA, llama.cpp runs a GGUF target with a
DFlash draft, vLLM with HyperQwen's patches its own 4-bit Qwen3.8-27B with a DFlash2 draft, and Lucebox a GGUF target
with a DFlash draft it verifies as a tree. All answer the same chat requests: task files, or the replay rows
compare/mixture_export.py writes. Tokens per forward pass counts generated tokens per target forward including prefill:
the mean accepted length.
"""

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Annotated

import typer
from bench import BenchRequest, BenchResponse, BenchSampling, ChatMessage
from engines import ROOT, prompt_tokens, run_engine
from server.main import get_bench_average
from server.util import load_input
from wsl import linux_command, linux_uv_command

app = typer.Typer(add_completion=False)

RESULTS = ROOT / "workspace/compare-results"
# performance-benchmarks sizes an automatic context as the prompt, the generation cap and this padding.
CONTEXT_PADDING = 64


class Engine(StrEnum):
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    LUCEBOX = "lucebox"


@app.command()
def main(
    engine: Annotated[
        list[Engine] | None, typer.Option(help="CUDA engine to compare with uzu; repeatable. Default: all.")
    ] = None,
    uzu_model: Annotated[str, typer.Option(help="uzu checkpoint.")] = "trymirai/Qwen3.8-27B-M",
    llamacpp_model: Annotated[
        str, typer.Option(help="llama.cpp 4-bit GGUF target: a path WSL can open or REPOSITORY:QUANT.")
    ] = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
    llamacpp_draft_model: Annotated[
        str, typer.Option(help="llama.cpp DFlash draft GGUF, in the same forms.")
    ] = "ggml-org/Qwen3.8-27B-GGUF:Q8_0",
    lucebox_model: Annotated[
        str, typer.Option(help="Lucebox 4-bit GGUF target, REPOSITORY:FILE.gguf.")
    ] = "Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF:Qwen3.8-27B-IQ4_XS-pure.gguf",
    lucebox_draft_model: Annotated[
        str, typer.Option(help="Lucebox DFlash draft GGUF, REPOSITORY:FILE.gguf.")
    ] = "Lucebox/Qwen3.8-27B-DFlash2-GGUF:Qwen3.8-27B-DFlash2-Q8_0.gguf",
    input: Annotated[
        list[Path] | None, typer.Option(help="Task JSON with 'messages'; repeatable. Default: london-summary.")
    ] = None,
    rows: Annotated[
        Path | None, typer.Option(help="Replay rows from compare/mixture_export.py, instead of tasks.")
    ] = None,
    num_runs: Annotated[int, typer.Option(min=1)] = 3,
    max_tokens: Annotated[int, typer.Option(min=2, help="Generation cap for tasks; rows carry their own.")] = 512,
    temperature: Annotated[float | None, typer.Option(help="Sampling temperature; greedy when omitted.")] = None,
    top_p: Annotated[float | None, typer.Option()] = None,
    top_k: Annotated[int | None, typer.Option()] = None,
    wait_cooldown: Annotated[
        bool, typer.Option("--wait-cooldown/--no-wait-cooldown", help="Wait for the GPU <= 60°C before measuring.")
    ] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for each response, including builds.")] = 3600,
) -> None:
    if input and rows:
        raise typer.BadParameter("Pass --input or --rows, not both")
    sampling = None if temperature is None else BenchSampling(temp=temperature, top_p=top_p, top_k=top_k)
    context: int | None = None
    if rows is not None:
        replay = [json.loads(line) for line in rows.read_text().splitlines() if line.strip()]
        labels = [f"{row['id'].rpartition(':')[2][:6]} {row['category']}" for row in replay]
        requests = [
            BenchRequest.model_validate(row["request"] | {"sampling": sampling, "num_runs": num_runs}) for row in replay
        ]
        # Engines that size their context at load get the largest row's.
        context = max(row["prompt_tokens"] + row["request"]["max_tokens"] for row in replay) + CONTEXT_PADDING
    else:
        tasks = input or [ROOT / "server/tasks/london-summary.json"]
        labels = [path.stem for path in tasks]
        requests = [
            BenchRequest(
                prompt_chat=[ChatMessage.model_validate(message) for message in load_input(path)["messages"]],
                max_tokens=max_tokens,
                sampling=sampling,
                num_runs=num_runs,
            )
            for path in tasks
        ]
    uzu_context = ["--context-length", str(context)] if context is not None else []
    lucebox_context = ["--max-ctx", str(context)] if context is not None else []
    commands = {
        "uzu": ["cargo", "run", "--release", "-p", "benchmarks-uzu", "--", "--model", uzu_model, *uzu_context],
        # llama.cpp sizes each run's context to its prompt and generation cap.
        Engine.LLAMACPP: linux_command(
            ["bash", "engine-llamacpp/run.sh", "--model", llamacpp_model, "--draft-model", llamacpp_draft_model]
        ),
        # HyperQwen serves 64K tokens of context beside its draft.
        Engine.VLLM: linux_uv_command("engine-vllm", "bench-vllm", ["--hyperqwen"]),
        Engine.LUCEBOX: linux_uv_command(
            "engine-lucebox",
            "bench-lucebox",
            ["--model", lucebox_model, "--draft-model", lucebox_draft_model, *lucebox_context],
        ),
    }

    output = RESULTS / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results: dict[str, list[list[BenchResponse]]] = {}
    for name in ["uzu", *(engine or Engine)]:
        try:
            results[name] = run_engine(name, commands[name], requests, wait_cooldown, timeout)
        except RuntimeError as error:
            typer.echo(f"{name} failed, continuing without it: {error}", err=True)
            continue
        save(output / f"{name}.json", labels, requests, results[name])

    print(
        f"{'engine':<8} {'request':<30} {'effort':<6} {'cap':>5} {'prompt':>6} {'output':>6} {'TTFT ms':>8} "
        f"{'decode t/s':>10} {'MAL':>5}"
    )
    for name, per_request in results.items():
        for label, request, responses in zip(labels, requests, per_request, strict=True):
            average = get_bench_average(responses)
            print(
                f"{name:<8} {label[:30]:<30} {request.reasoning_effort or '-':<6} {request.max_tokens:>5} "
                f"{prompt_tokens(responses[0]):>6} {fmean(r.tokens_count for r in responses):>6.0f} "
                f"{average.time_to_first_token * 1000:>8.1f} {average.decode_tps:>10.1f} "
                f"{average.tokens_per_forward_pass:>5.2f}"
            )
    averages = {}
    for name, per_request in results.items():
        responses = [response for responses in per_request for response in responses]
        averages[name] = average = get_bench_average(responses)
        decoded = [response for response in responses if response.decode_tps > 0]
        weighted = sum(r.tokens_count - 1 for r in decoded) / sum((r.tokens_count - 1) / r.decode_tps for r in decoded)
        print(
            f"{name}: decode {average.decode_tps:.1f} t/s (mean over runs), {weighted:.1f} t/s (token-weighted), "
            f"MAL {average.tokens_per_forward_pass:.2f}, TTFT {average.time_to_first_token * 1000:.1f} ms, "
            f"prefill {average.prompt_tps:.0f} t/s"
        )
    if (uzu := averages.get("uzu")) is not None:
        for name, other in averages.items():
            if name != "uzu":
                print(
                    f"uzu / {name}: decode {uzu.decode_tps / other.decode_tps:.2f}x, "
                    f"MAL {uzu.tokens_per_forward_pass / other.tokens_per_forward_pass:.2f}x"
                )
    if results:
        typer.echo(f"Results without generated text: {output}", err=True)


def save(path: Path, labels: list[str], requests: list[BenchRequest], results: list[list[BenchResponse]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "request": label,
            "reasoning_effort": request.reasoning_effort,
            "max_tokens": request.max_tokens,
            "responses": [response.model_dump(exclude={"text"}) for response in responses],
        }
        for label, request, responses in zip(labels, requests, results, strict=True)
    ]
    path.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    app()
