import sys
import time
from collections.abc import Callable, Generator
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Annotated, cast

import mlx_lm
import typer
from bench import BenchRequest, BenchResponse, ModelConfig
from common import (
    InferenceEngine,
    check_specdec,
    get_finish_reason,
    get_model_path,
    get_sampling,
    get_tokenized_prompt,
    run_loop,
)
from mach import MemoryCounters, get_gpu_energy, get_memory_counters
from mlx import core as mx
from mlx import nn
from mlx_lm.generate import GenerationResponse
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper


class MLXConfig(ModelConfig):
    revision: str | None = None


class MLXEngine(InferenceEngine):
    model: nn.Module
    tokenizer: TokenizerWrapper

    def __init__(self, config: MLXConfig):
        self.sampling = get_sampling(config.sampling, supports_repetition=True)
        check_specdec(config, False)
        super().__init__()
        model_path = get_model_path(config.model, config.revision)
        self.model, self.tokenizer = cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(model_path))

    def execute(self, request: BenchRequest) -> BenchResponse:
        prompt: list[int] = get_tokenized_prompt(request, self.tokenizer)
        if request.seed is not None:
            mx.random.seed(request.seed)

        sampling = self.sampling
        sampler = make_sampler(
            temp=1.0 if sampling.temp is None else sampling.temp,
            top_p=1.0 if sampling.top_p is None else sampling.top_p,
            min_p=0.0 if sampling.min_p is None else sampling.min_p,
            min_tokens_to_keep=1,
            top_k=0 if sampling.top_k is None else sampling.top_k,
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
            max_tokens=request.max_output_tokens or -1,
            sampler=sampler,
            logits_processors=make_logits_processors(
                repetition_penalty=sampling.repetition_penalty,
                repetition_context_size=sampling.suffix_repetition_length,
            ),
        )

        return self._run(generate)

    def _run(self, generate: Callable[..., Generator[GenerationResponse]]) -> BenchResponse:
        # prepare variables
        text: str = ""
        time_to_first_token: float = -1.0
        response: GenerationResponse | None = None
        mem_counters_max: MemoryCounters = get_memory_counters()

        def update_memory(*_progress: int) -> None:
            nonlocal mem_counters_max
            counters = get_memory_counters()
            if counters.graphics_total > mem_counters_max.graphics_total:
                mem_counters_max = counters

        # create and run inference loop
        mx.reset_peak_memory()
        energy_start = get_gpu_energy()
        time_start: float = time.perf_counter()
        stream: Generator[GenerationResponse] = generate(prompt_progress_callback=update_memory)
        for response in stream:
            if time_to_first_token < 0.0:
                time_to_first_token = time.perf_counter() - time_start
            text += response.text

            update_memory()
        time_total: float = time.perf_counter() - time_start
        energy_gpu = get_gpu_energy() - energy_start

        if response is None:
            raise RuntimeError("Generation did not return a response")

        # The first token is produced during prefill, outside the decode interval.
        decode_duration = time_total - time_to_first_token
        decode_tokens = max(0, response.generation_tokens - 1)

        return BenchResponse(
            text=text,
            finish_reason=get_finish_reason(response.finish_reason),
            input_tokens_count=response.prompt_tokens,
            output_tokens_count=response.generation_tokens,
            time_to_first_token=time_to_first_token,
            prefill_tps=response.prompt_tps,
            decode_tps=decode_tokens / decode_duration if decode_duration > 0.0 else 0.0,
            tokens_per_forward_pass=None,
            duration=time_total,
            energy_gpu=energy_gpu,
            memory_used=mx.get_peak_memory(),
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident=mem_counters_max.resident_size,
            memory_graphics_total=mem_counters_max.graphics_total,
        )


def run(model: Annotated[Path, typer.Argument(help="Engine model configuration JSON.")]) -> None:
    engine: MLXEngine
    with redirect_stdout(sys.stderr):
        engine = MLXEngine(MLXConfig.model_validate_json(model.read_text()))
    run_loop(engine)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
