"""Compare uzu and vLLM on one NVIDIA GPU on Windows, each with its own speculative decoding.

uzu runs a trymirai checkpoint with its speculator on Vulkan. vLLM runs a Hugging Face target with a DFlash draft on
CUDA inside WSL. Both decode the same chat requests greedily. Tokens per forward pass counts generated tokens per target
forward including prefill: the mean accepted length.
"""

from pathlib import Path
from statistics import fmean
from typing import Annotated

import typer
from bench import BenchRequest, ChatMessage
from engines import ROOT, prompt_tokens, run_engine
from server.main import get_bench_average
from server.util import load_input
from wsl import linux_uv_command

app = typer.Typer(add_completion=False)


@app.command()
def main(
    vllm_model: Annotated[str, typer.Option(help="vLLM 4-bit target.")] = "Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM",
    uzu_model: Annotated[str, typer.Option(help="uzu checkpoint.")] = "trymirai/Qwen3.8-27B-M",
    vllm_draft_model: Annotated[str, typer.Option(help="vLLM DFlash draft.")] = "syvai/Qwen3.8-27B-DFlash2-W4A16",
    input: Annotated[
        list[Path] | None, typer.Option(help="Task JSON with 'messages'; repeatable. Default: london-summary.")
    ] = None,
    num_runs: Annotated[int, typer.Option(min=1)] = 3,
    max_tokens: Annotated[int, typer.Option(min=2)] = 512,
    wait_cooldown: Annotated[
        bool, typer.Option("--wait-cooldown/--no-wait-cooldown", help="Wait for the GPU <= 60°C before measuring.")
    ] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for each response, including builds.")] = 3600,
) -> None:
    input = input or [ROOT / "server/tasks/london-summary.json"]
    requests = [
        BenchRequest(
            prompt_chat=[ChatMessage.model_validate(message) for message in load_input(path)["messages"]],
            max_tokens=max_tokens,
            num_runs=num_runs,
        )
        for path in input
    ]
    engines = {
        "uzu": ["cargo", "run", "--release", "-p", "benchmarks-uzu", "--", "--model", uzu_model],
        # A 27B 4-bit target and its draft leave a 24 GB GPU too little KV memory for vLLM's default context, so each
        # prompt plus --max-tokens must fit in 2048 tokens.
        "vllm": linux_uv_command(
            "engine-vllm",
            "bench-vllm",
            ["--model", vllm_model, "--draft-model", vllm_draft_model, "--max-model-len", "2048"],
        ),
    }
    results = {name: run_engine(name, command, requests, wait_cooldown, timeout) for name, command in engines.items()}

    print(f"{'engine':<6} {'task':<28} {'prompt':>6} {'output':>6} {'TTFT ms':>8} {'decode t/s':>10} {'MAL':>5}")
    averages = {}
    for name, per_request in results.items():
        for path, responses in zip(input, per_request, strict=True):
            average = get_bench_average(responses)
            print(
                f"{name:<6} {path.stem[:28]:<28} {prompt_tokens(responses[0]):>6} "
                f"{fmean(r.tokens_count for r in responses):>6.0f} {average.time_to_first_token * 1000:>8.1f} "
                f"{average.decode_tps:>10.1f} {average.tokens_per_forward_pass:>5.2f}"
            )
        averages[name] = get_bench_average([response for responses in per_request for response in responses])
    for name, average in averages.items():
        print(
            f"{name}: decode {average.decode_tps:.1f} t/s, MAL {average.tokens_per_forward_pass:.2f}, "
            f"TTFT {average.time_to_first_token * 1000:.1f} ms"
        )
    uzu, vllm = averages["uzu"], averages["vllm"]
    print(
        f"uzu / vLLM: decode {uzu.decode_tps / vllm.decode_tps:.2f}x, "
        f"MAL {uzu.tokens_per_forward_pass / vllm.tokens_per_forward_pass:.2f}x"
    )


if __name__ == "__main__":
    app()
