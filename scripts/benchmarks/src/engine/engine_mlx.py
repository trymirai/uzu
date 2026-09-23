import time
from collections.abc import Callable, Generator
from functools import partial
from pathlib import Path
from typing import cast

import mlx_lm
from mlx import nn
from mlx_lm.generate import GenerationResponse
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from bench import BenchRequest, BenchResponse
from common import get_model_path
from mach import MemoryCounters, get_memory_counters


def _run_single(generate: Callable[[], Generator[GenerationResponse]]) -> BenchResponse:
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


def run(
    model: str | Path,
    config: BenchRequest,
    *,
    prefill_step_size: int | None = None,
    draft_model_path: str | Path | None = None,
) -> list[BenchResponse]:
    num_runs = config.num_runs if config.num_runs is not None else 1
    if num_runs < 1:
        raise ValueError("num_runs must be 1 or greater")

    # load main model
    model_path: str = get_model_path(model)
    mlx_model: nn.Module
    tokenizer: TokenizerWrapper
    mlx_model, tokenizer = cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(model_path))

    # load draft model
    draft_model: nn.Module | None = None
    if draft_model_path is not None:
        draft_path: str = get_model_path(draft_model_path)
        draft_tokenizer: TokenizerWrapper
        draft_model, draft_tokenizer = cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(draft_path))
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match target tokenizer")

    # prepare prompt
    prompt: str | list[int]
    if isinstance(config.prompt, str):
        prompt = config.prompt
    else:
        messages = [message.model_dump(mode="json") for message in config.prompt]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    # sampling
    sampler: Callable | None = None
    if config.sampling is not None:
        sampler = make_sampler(
            temp=config.sampling.temp or 0.0,
            top_p=config.sampling.top_p or 0.0,
            min_p=config.sampling.min_p or 0.0,
            min_tokens_to_keep=1,
            top_k=config.sampling.top_k or 0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
            xtc_special_tokens=[],
        )

    generation_options: dict[str, int] = {}
    if prefill_step_size is not None:
        generation_options["prefill_step_size"] = prefill_step_size
    if config.speculative_depth is not None:
        generation_options["num_draft_tokens"] = config.speculative_depth
    generate = partial(
        mlx_lm.stream_generate,
        model=mlx_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=config.max_tokens or 256,
        draft_model=draft_model,
        sampler=sampler,
        **generation_options,
    )

    return [_run_single(generate) for _ in range(num_runs)]
