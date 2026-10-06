import json
from pathlib import Path
from statistics import fmean
from typing import Annotated, Any, cast

import typer
from bench import BenchResponse
from openai import OpenAI, omit
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from .engine import ServerEngine, ServerEngineType
from .engine.uzu import UzuServerEngine
from .util import await_cooldown, get_openai_base_url, load_input

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
    )


@app.command(name="engine")
def run_engine(
    engine_type: ServerEngineType,
    model: Annotated[str, typer.Option("--model", help="Model identifier, repository ID, or local model directory.")],
    input: Annotated[Path, typer.Option("--input", help="Path to a JSON file.")],
    num_runs: Annotated[int, typer.Option(min=1, help="Number of times to send the request sequentially.")] = 1,
    wait_cooling: Annotated[
        bool, typer.Option("--wait-cooling/--no-wait-cooling", help="Wait before CPU and GPU are cooled enough.")
    ] = True,
) -> None:
    request: dict[str, Any] = load_input(input)

    engine: ServerEngine
    if engine_type == ServerEngineType.UZU:
        engine = UzuServerEngine(DEFAULT_HOST, DEFAULT_PORT, model)
    else:
        raise ValueError(f"Engine {engine_type} is not supported")

    engine.start()
    responses: list[BenchResponse] = []
    try:
        for _ in range(num_runs):
            if wait_cooling:
                await_cooldown()
            response = engine.handle_request(request)
            responses.append(response)
    except Exception as error:
        typer.echo(f"Exception: {error}")
    engine.stop()

    average = get_bench_average(responses)
    typer.echo(json.dumps(average.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command(name="common")
def run_server(
    input: Annotated[Path, typer.Option("--input", help="Path to a JSON file.")],
    model: Annotated[str | None, typer.Option("--model", help="Model name. Omit to use the server default.")] = None,
    host: Annotated[str, typer.Option(help="Server hostname or IP address.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(min=1, max=65535, help="Server HTTP port.")] = DEFAULT_PORT,
    num_runs: Annotated[int, typer.Option(min=1, help="Number of times to send the request sequentially.")] = 1,
    wait_cooling: Annotated[
        bool, typer.Option("--wait-cooling/--no-wait-cooling", help="Wait before CPU and GPU are cooled enough.")
    ] = True,
) -> None:
    request: dict[str, Any] = load_input(input)
    responses: list[dict[str, Any]] = []

    with OpenAI(
        base_url=get_openai_base_url(host, port),
        api_key="not-needed",
    ) as client:
        for _ in range(num_runs):
            if wait_cooling:
                await_cooldown()
            response: ChatCompletion = client.chat.completions.create(
                model=model if model is not None else cast(Any, omit),
                messages=request["messages"],
            )
            response_json: dict[str, Any] = response.model_dump(mode="json")
            responses.append(response_json)

    typer.echo(json.dumps(responses, ensure_ascii=False, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
