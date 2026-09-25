import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from bench import BenchRequest, BenchResponse
from common import InferenceEngine, get_model_path, run_loop
from mach import MemoryCounters, get_memory_counters
from mtplx.generation import GenerationOutput, generate_mtpk
from mtplx.runtime import MTPLXRuntime, load
from mtplx.sampling import SamplerConfig


class MTPLXEngine(InferenceEngine):
    runtime: MTPLXRuntime

    def __init__(self, model: str | Path):
        super().__init__()
        model_path: str = get_model_path(model)
        self.runtime = load(model_path)

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        num_runs = request.num_runs or 1
        if num_runs < 1:
            raise ValueError("num_runs must be 1 or greater")

        # prepare prompt
        prompt_ids: list[int]
        if isinstance(request.prompt, str):
            prompt_ids = self.runtime.tokenizer.encode(request.prompt)
        else:
            messages = [message.model_dump(mode="json") for message in request.prompt]
            prompt_ids = self.runtime.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **request.model_dump(mode="json", include={"tools", "tool_choice"}),
            )

        sampler: SamplerConfig
        if request.sampling is None:
            sampler = SamplerConfig()
        else:
            sampler = SamplerConfig(
                temperature=request.sampling.temp or 0.6,
                top_p=request.sampling.top_p or 0.95,
                top_k=request.sampling.top_k or 20,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )

        generate = partial(
            generate_mtpk,
            self.runtime,
            prompt_ids,
            max_tokens=request.max_tokens or 256,
            sampler=sampler,
            speculative_depth=request.speculative_depth or 3,
        )
        return [self._run(generate) for _ in range(num_runs)]

    def _run(self, generate: Callable[..., GenerationOutput]) -> BenchResponse:
        # prepare variables
        time_first_token: float = -1.0
        mem_graphics_max: int = 0
        mem_counters_max: MemoryCounters = get_memory_counters()

        def token_callback(token_ids: list[int]) -> None:
            nonlocal time_first_token, mem_graphics_max, mem_counters_max
            if token_ids and time_first_token < 0.0:
                time_first_token = time.perf_counter()

            mem_counters = get_memory_counters()
            if mem_counters.graphics_total > mem_graphics_max:
                mem_graphics_max = mem_counters.graphics_total
                mem_counters_max = mem_counters

        # create and run inference
        time_start: float = time.perf_counter()
        output = generate(token_callback=token_callback)
        time_total: float = time.perf_counter() - time_start

        # collect metrics
        forward_passes = int(output.stats.verify_calls)
        if forward_passes == 0:
            raise RuntimeError("Forward passes is 0")
        tokens_per_forward_pass = max(0, output.stats.generated_tokens - 1) / forward_passes

        return BenchResponse(
            text=output.text,
            time_to_first_token=(time_first_token - time_start),
            prompt_tps=output.stats.prompt_tps,
            decode_tps=output.stats.decode_tok_s,
            tokens_per_forward_pass=tokens_per_forward_pass,
            duration=time_total,
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident_peak=mem_counters_max.resident_size_peak,
            memory_graphics_total=mem_counters_max.graphics_total,
        )


def run(model: Annotated[str, typer.Option("-m", "--model")]) -> None:
    engine = MTPLXEngine(model)
    run_loop(engine)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
