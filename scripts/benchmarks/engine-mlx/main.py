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
from common import InferenceEngine, get_model_path, run_loop
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
        model_path: str = get_model_path(model)
        self.model, self.tokenizer = cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(model_path))

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        num_runs = request.num_runs if request.num_runs is not None else 1
        if num_runs < 1:
            raise ValueError("num_runs must be 1 or greater")

        # prepare prompt
        prompt: str | list[int]
        if isinstance(request.prompt, str):
            prompt = request.prompt
        else:
            messages = [message.model_dump(mode="json") for message in request.prompt]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **request.model_dump(mode="json", include={"tools", "tool_choice"}),
            )

        # sampling
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

        generation_options: dict[str, int] = {}
        if request.speculative_depth is not None:
            generation_options["num_draft_tokens"] = request.speculative_depth
        generate = partial(
            mlx_lm.stream_generate,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens or 256,
            sampler=sampler,
            **generation_options,
        )

        return [self._run(generate) for _ in range(num_runs)]

    def _run(self, generate: Callable[[], Generator[GenerationResponse]]) -> BenchResponse:
        # prepare variables
        text: str = ""
        time_to_first_token: float = -1.0
        draft_flags: list[bool] = []
        response: GenerationResponse | None = None
        mem_graphics_max: int = 0
        mem_counters_max: MemoryCounters = get_memory_counters()

        # create and run inference loop
        time_start: float = time.perf_counter()
        stream: Generator[GenerationResponse] = generate()
        for response in stream:
            if time_to_first_token < 0.0:
                time_to_first_token = time.perf_counter() - time_start
            text += response.text
            draft_flags.append(response.from_draft)

            mem_counters = get_memory_counters()
            if mem_counters.graphics_total > mem_graphics_max:
                mem_graphics_max = mem_counters.graphics_total
                mem_counters_max = mem_counters
        time_total: float = time.perf_counter() - time_start

        if response is None:
            raise RuntimeError("Generation did not return a response")

        # Collect tokens per forward passes
        target_forward_passes: int = sum(not flag for flag in draft_flags)
        if draft_flags and draft_flags[-1]:
            # A trailing draft group also used one target verification pass.
            target_forward_passes += 1
        tokens_per_forward_pass: float = response.generation_tokens / target_forward_passes

        return BenchResponse(
            text=text,
            time_to_first_token=time_to_first_token,
            prompt_tps=response.prompt_tps,
            decode_tps=response.generation_tps,
            tokens_per_forward_pass=tokens_per_forward_pass,
            duration=time_total,
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident_peak=mem_counters_max.resident_size_peak,
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
