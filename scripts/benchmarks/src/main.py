from pathlib import Path
from typing import Annotated

from pydantic import TypeAdapter
from typer import BadParameter, Option, Typer

from bench import BenchRequest, BenchResponse
from common import Engine
from engine import engine_llamacpp, engine_mlx, engine_mtplx

app = Typer(add_completion=False)


@app.command()
def run(
    engine: Annotated[Engine, Option("-e", "--engine")],
    model: Annotated[str, Option("-m", "--model")],
    input_path: Annotated[Path, Option("-i", "--input", exists=True, dir_okay=False, readable=True)],
    output_path: Annotated[Path | None, Option("-o", "--output", dir_okay=False)] = None,
):
    # prepare input
    input_config: BenchRequest
    try:
        input_config = BenchRequest.model_validate_json(input_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BadParameter(str(error), param_hint="--input") from error

    # run inference
    output: list[BenchResponse]
    match engine:
        case Engine.LLAMA_CPP:
            output = engine_llamacpp.run(model, input_path)
        case Engine.MLX:
            output = engine_mlx.run(model, input_config)
        case Engine.MTPLX:
            output = engine_mtplx.run(model, input_config)
        case _:
            raise BadParameter(f"Engine '{engine.value}' is not supported", param_hint="--engine")

    # handle output
    output_json = TypeAdapter(list[BenchResponse]).dump_json(output, indent=2).decode("utf-8")
    if output_path is None:
        print(output_json)
    else:
        try:
            output_path.write_text(output_json + "\n", encoding="utf-8")
        except OSError as error:
            raise BadParameter(str(error), param_hint="--output") from error


if __name__ == "__main__":
    app()
