import asyncio
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from bench import BenchRequest, BenchResponse, BenchSampling
from common import InferenceEngine, get_model_path, get_tokenized_prompt, run_loop
from mach import MemoryCounters, get_memory_counters
from omlx.engine.base import GenerationOutput
from omlx.engine.batched import BatchedEngine
from omlx.model_settings import ModelSettings


class OMLXEngine(InferenceEngine):
    engine: BatchedEngine
    runner: asyncio.Runner
    closed: bool

    def __init__(self, model: str | Path):
        super().__init__()
        model_path = get_model_path(model)
        self.engine = BatchedEngine(model_name=model_path, model_settings=ModelSettings())
        self.runner = asyncio.Runner()
        self.closed = False
        try:
            self.runner.run(self.engine.start())
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True
        try:
            self.runner.run(self.engine.stop())
        finally:
            self.runner.close()

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        if self.closed:
            raise RuntimeError("Engine is closed")

        return self.runner.run(self._execute(request))

    async def _execute(self, request: BenchRequest) -> list[BenchResponse]:
        num_runs = request.num_runs if request.num_runs is not None else 1
        if num_runs < 1:
            raise ValueError("num_runs must be 1 or greater")

        if request.speculative_depth not in (None, 0):
            raise ValueError("OMLXEngine does not support speculative decoding")

        prompt: list[int] = get_tokenized_prompt(request, self.engine.tokenizer)

        sampling = request.sampling or BenchSampling()
        generate = partial(
            self.engine.stream_generate,
            prompt=prompt,
            max_tokens=request.max_tokens or 256,
            temperature=sampling.temp if sampling.temp is not None else 0.7,
            top_p=sampling.top_p if sampling.top_p is not None else 0.9,
            min_p=sampling.min_p or 0.0,
            top_k=sampling.top_k or 0,
        )

        return [await self._run(generate) for _ in range(num_runs)]

    async def _run(self, generate: Callable[[], AsyncIterator[GenerationOutput]]) -> BenchResponse:
        time_first_token: float | None = None
        time_last_token: float | None = None
        completion_tokens = 0
        response: GenerationOutput | None = None
        mem_counters_max: MemoryCounters = get_memory_counters()

        time_start: float = time.perf_counter()
        async for response in generate():
            if response.completion_tokens > completion_tokens:
                generated_at = response.generated_at
                if generated_at is None:
                    generated_at = time.perf_counter()
                if time_first_token is None:
                    time_first_token = generated_at
                time_last_token = response.generated_until if response.generated_until is not None else generated_at
                completion_tokens = response.completion_tokens

            mem_counters = get_memory_counters()
            if mem_counters.graphics_total > mem_counters_max.graphics_total:
                mem_counters_max = mem_counters
        time_total: float = time.perf_counter() - time_start

        if response is None:
            raise RuntimeError("Generation did not return a response")

        # An immediate stop can finish without producing any completion tokens.
        time_to_first_token = time_first_token - time_start if time_first_token is not None else time_total
        decode_time = (
            time_last_token - time_first_token if time_first_token is not None and time_last_token is not None else 0.0
        )
        prompt_tokens = response.prompt_tokens - response.cached_tokens

        return BenchResponse(
            text=response.text,
            time_to_first_token=time_to_first_token,
            prompt_tps=prompt_tokens / time_to_first_token if time_to_first_token > 0.0 else 0.0,
            decode_tps=(completion_tokens - 1) / decode_time if decode_time > 0.0 else 0.0,
            tokens_per_forward_pass=1.0 if completion_tokens else 0.0,
            duration=time_total,
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident=mem_counters_max.resident_size,
            memory_graphics_total=mem_counters_max.graphics_total,
        )


def run(model: Annotated[str, typer.Option("-m", "--model")]) -> None:
    engine: OMLXEngine
    with redirect_stdout(sys.stderr):
        engine = OMLXEngine(Path(model) if Path(model).expanduser().is_dir() else model)
    try:
        run_loop(engine)
    finally:
        with redirect_stdout(sys.stderr):
            engine.close()


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
