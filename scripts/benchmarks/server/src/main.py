import json
from pathlib import Path
from statistics import fmean
from typing import Annotated, Any

import typer
from bench import BenchResponse
from pydantic import BaseModel

from .engine import ServerEngine, ServerEngineType
from .engine.magnitude import MagnitudeServerEngine
from .engine.splash import SplashServerEngine
from .engine.tensorfold import TensorFoldServerEngine
from .engine.uzu import UzuServerEngine
from .util import await_cooldown, load_input

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

app = typer.Typer(add_completion=False)


class BenchAverage(BaseModel):
    time_to_first_token: float
    prompt_tps: float
    decode_tps: float
    tokens_per_forward_pass: float
    duration: float
    max_memory_phys_footprint: int
    max_memory_memory_resident: int
    max_memory_graphics_total: int
    number_of_runs: int


def get_bench_average(responses: list[BenchResponse]) -> BenchAverage:
    if not responses:
        raise ValueError("Cannot average an empty list of benchmark responses.")

    max_memory_response = max(responses, key=lambda response: response.memory_graphics_total)
    return BenchAverage(
        time_to_first_token=fmean(response.time_to_first_token for response in responses),
        prompt_tps=fmean(response.prompt_tps for response in responses),
        decode_tps=fmean(response.decode_tps for response in responses),
        tokens_per_forward_pass=fmean(response.tokens_per_forward_pass for response in responses),
        duration=fmean(response.duration for response in responses),
        max_memory_phys_footprint=max_memory_response.memory_phys_footprint,
        max_memory_memory_resident=max_memory_response.memory_resident,
        max_memory_graphics_total=max_memory_response.memory_graphics_total,
        number_of_runs=len(responses),
    )


@app.command(name=ServerEngineType.MAGNITUDE.value)
@app.command(name=ServerEngineType.TENSORFOLD.value)
@app.command(name=ServerEngineType.SPLASH.value)
@app.command(name=ServerEngineType.UZU.value)
def run_engine(
    ctx: typer.Context,
    model: Annotated[str, typer.Option("--model", help="Model identifier, repository ID, or local model directory.")],
    input: Annotated[Path, typer.Option("--input", help="Path to a JSON file.")],
    num_runs: Annotated[int, typer.Option(min=1, help="Number of times to send the request sequentially.")] = 1,
    wait_cooling: Annotated[
        bool,
        typer.Option(
            "--wait-cooldown/--no-wait-cooldown", help="Wait until CPU and GPU are at or below 60°C before each run."
        ),
    ] = True,
) -> None:
    engine_type = ServerEngineType(ctx.info_name)
    request: dict[str, Any] = load_input(input)

    engine: ServerEngine
    if engine_type == ServerEngineType.UZU:
        engine = UzuServerEngine(DEFAULT_HOST, DEFAULT_PORT, model)
    elif engine_type == ServerEngineType.TENSORFOLD:
        engine = TensorFoldServerEngine(DEFAULT_HOST, DEFAULT_PORT, model)
    elif engine_type == ServerEngineType.MAGNITUDE:
        engine = MagnitudeServerEngine(DEFAULT_HOST, DEFAULT_PORT, model)
    elif engine_type == ServerEngineType.SPLASH:
        engine = SplashServerEngine(DEFAULT_HOST, DEFAULT_PORT, model)
    else:
        raise ValueError(f"Engine {engine_type} is not supported")

    responses: list[BenchResponse] = []
    try:
        typer.echo(f"Starting {engine_type.value} server...", err=True)
        engine.start()
        typer.echo("Server ready.", err=True)
        for run in range(1, num_runs + 1):
            if wait_cooling:
                await_cooldown()
            typer.echo(f"Benchmark run {run}/{num_runs}...", err=True)
            response = engine.handle_request(request)
            responses.append(response)
            typer.echo(f"Benchmark run {run}/{num_runs} completed in {response.duration:.2f}s.", err=True)
    finally:
        engine.stop()

    average = get_bench_average(responses)
    typer.echo(json.dumps(average.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
