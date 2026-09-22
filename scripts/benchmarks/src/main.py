from pathlib import Path
from typing import Annotated

from typer import BadParameter, Option, Typer

import engine_llamacpp
import engine_mlx
import engine_mtplx
from common import BenchInput, BenchOutput, Engine
from engine_mlx import MlxRunRequest, MlxRunResponse, MlxSampling
from engine_mtplx import MtplxRunRequest, MtplxRunResponse, MtplxSampling

app = Typer(add_completion=False)


def run_llamacpp(model: str | Path, input_path: Path) -> BenchOutput:
    return engine_llamacpp.run(model, input_path)


def run_mlx(model: str | Path, config: BenchInput) -> BenchOutput:
    sampling: MlxSampling | None = None
    if config.sampling is not None:
        sampling = MlxSampling(
            temp=config.sampling.temp,
            top_p=config.sampling.top_p,
            min_p=config.sampling.min_p,
            top_k=config.sampling.top_k,
        )
    request = MlxRunRequest(
        model=model,
        prompt=config.prompt,
        max_tokens=config.max_tokens,
        sampling=sampling,
    )

    response: MlxRunResponse = engine_mlx.run(request)

    return BenchOutput(
        text=response.text,
        time_to_first_token=response.time_to_first_token,
        prompt_tps=response.prompt_tps,
        decode_tps=response.generation_tps,
        tokens_per_forward_pass=response.tokens_per_fp,
        duration=response.duration,
        memory_phys_footprint=response.memory_counters.phys_footprint,
        memory_resident_peak=response.memory_counters.resident_size_peak,
        memory_graphics_total=response.memory_counters.graphics_total,
    )


def run_mtplx(model: str | Path, config: BenchInput) -> BenchOutput:
    sampling: MtplxSampling | None = None
    if config.sampling is not None:
        sampling = MtplxSampling(
            temperature=config.sampling.temp,
            top_p=config.sampling.top_p,
            top_k=config.sampling.top_k,
            presence_penalty=None,
            frequency_penalty=None,
        )
    request = MtplxRunRequest(
        model=model,
        prompt=config.prompt,
        max_tokens=config.max_tokens,
        speculative_depth=config.speculative_depth,
        sampling=sampling,
    )

    response: MtplxRunResponse = engine_mtplx.run(request)

    return BenchOutput(
        text=response.text,
        time_to_first_token=response.time_to_first_token,
        prompt_tps=response.prompt_tps,
        decode_tps=response.generation_tps,
        tokens_per_forward_pass=response.tokens_per_fp,
        duration=response.duration,
        memory_phys_footprint=response.memory_counters.phys_footprint,
        memory_resident_peak=response.memory_counters.resident_size_peak,
        memory_graphics_total=response.memory_counters.graphics_total,
    )


@app.command()
def run(
    engine: Annotated[Engine, Option("-e", "--engine")],
    model: Annotated[str, Option("-m", "--model")],
    input_path: Annotated[Path, Option("-i", "--input", exists=True, dir_okay=False, readable=True)],
    output_path: Annotated[Path | None, Option("-o", "--output", dir_okay=False)] = None,
):
    # prepare input
    input_config: BenchInput
    try:
        input_config = BenchInput.model_validate_json(input_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BadParameter(str(error), param_hint="--input") from error

    # run inference
    output: BenchOutput
    match engine:
        case Engine.LLAMA_CPP:
            output = run_llamacpp(model, input_path)
        case Engine.MLX:
            output = run_mlx(model, input_config)
        case Engine.MTPLX:
            output = run_mtplx(model, input_config)
        case _:
            raise BadParameter(f"Engine '{engine.value}' is not supported", param_hint="--engine")

    # handle output
    output_json = output.model_dump_json(indent=4)
    if output_path is None:
        print(output_json)
    else:
        try:
            output_path.write_text(output_json + "\n", encoding="utf-8")
        except OSError as error:
            raise BadParameter(str(error), param_hint="--output") from error


if __name__ == "__main__":
    app()
