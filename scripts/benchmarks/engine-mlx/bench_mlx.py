import sys
import time
from collections.abc import Callable, Generator
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Annotated, cast

import mlx_lm
import typer
from bench import BenchRequest, BenchResponse
from common import InferenceEngine, get_model_path, get_tokenized_prompt, run_loop
from mach import MemoryCounters, get_memory_counters
from mlx import nn
from mlx_lm.generate import GenerationResponse
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper


class MLXEngine(InferenceEngine):
    model: nn.Module
    tokenizer: TokenizerWrapper

    def __init__(self, model: str | Path):
        super().__init__()
        local_path = Path(model).expanduser()
        model_path = str(local_path) if local_path.is_dir() else get_model_path(model)
        self.model, self.tokenizer = cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(model_path))

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        num_runs = request.num_runs if request.num_runs is not None else 1
        if num_runs < 1:
            raise ValueError("num_runs must be 1 or greater")
        if request.speculative_depth not in (None, 0):
            raise ValueError("MLXEngine does not support speculative decoding")

        prompt: list[int] = get_tokenized_prompt(request, self.tokenizer)

        sampler: Callable | None = None
        if request.sampling is not None:
            sampler = make_sampler(
                temp=request.sampling.temp or 0.0,
                top_p=request.sampling.top_p or 0.0,
                min_p=request.sampling.min_p or 0.0,
                min_tokens_to_keep=1,
                top_k=request.sampling.top_k or 0,
                xtc_probability=0.0,
                xtc_threshold=0.0,
                xtc_special_tokens=[],
            )

        generate = partial(
            mlx_lm.stream_generate,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            # MLX uses -1 for unlimited generation; EOS still ends the stream.
            max_tokens=request.max_tokens or -1,
            sampler=sampler,
        )

        return [self._run(generate) for _ in range(num_runs)]

    def _run(self, generate: Callable[..., Generator[GenerationResponse]]) -> BenchResponse:
        # prepare variables
        text: str = ""
        time_to_first_token: float = -1.0
        response: GenerationResponse | None = None
        mem_counters_max: MemoryCounters = get_memory_counters()

        def update_memory(*_progress: int) -> None:
            nonlocal mem_counters_max
            counters = get_memory_counters()
            if counters.resident_size > mem_counters_max.resident_size:
                mem_counters_max = counters

        # create and run inference loop
        time_start: float = time.perf_counter()
        stream: Generator[GenerationResponse] = generate(prompt_progress_callback=update_memory)
        for response in stream:
            if time_to_first_token < 0.0:
                time_to_first_token = time.perf_counter() - time_start
            text += response.text

            update_memory()
        time_total: float = time.perf_counter() - time_start

        if response is None:
            raise RuntimeError("Generation did not return a response")

        # The first token is produced during prefill, outside the decode interval.
        decode_duration = time_total - time_to_first_token
        decode_tokens = max(0, response.generation_tokens - 1)

        return BenchResponse(
            text=text,
            tokens_count=response.generation_tokens,
            time_to_first_token=time_to_first_token,
            prompt_tps=response.prompt_tps,
            decode_tps=decode_tokens / decode_duration if decode_duration > 0.0 else 0.0,
            tokens_per_forward_pass=1.0,
            duration=time_total,
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident=mem_counters_max.resident_size,
            memory_graphics_total=mem_counters_max.graphics_total,
        )


def run(model: Annotated[str, typer.Option("-m", "--model")]) -> None:
    engine: MLXEngine
    with redirect_stdout(sys.stderr):
        engine = MLXEngine(model)
    run_loop(engine)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
