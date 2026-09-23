import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import huggingface_hub
from mtplx.generation import GenerationOutput, generate_mtpk
from mtplx.runtime import MTPLXRuntime, load
from mtplx.sampling import SamplerConfig

from bench import BenchRequest, BenchResponse
from mach import MemoryCounters, get_memory_counters


def _run_single(generate: Callable[..., GenerationOutput]) -> BenchResponse:
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


def run(model: str | Path, config: BenchRequest) -> list[BenchResponse]:
    num_runs = config.num_runs or 1
    if num_runs < 1:
        raise ValueError("num_runs must be 1 or greater")

    # load model
    model_path: str
    if isinstance(model, Path):
        model_path = model.expanduser().as_posix()
    else:
        model_path = huggingface_hub.snapshot_download(
            repo_id=model,
            cache_dir=Path("~/.cache/huggingface/hub").expanduser(),
            local_files_only=False,
        )
    runtime: MTPLXRuntime = load(model_path)

    # prepare prompt
    prompt_ids: list[int]
    if isinstance(config.prompt, str):
        prompt_ids = runtime.tokenizer.encode(config.prompt)
    else:
        messages = [{"role": message.role.value, "content": message.message} for message in config.prompt]
        prompt_ids = runtime.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    sampler: SamplerConfig
    if config.sampling is None:
        sampler = SamplerConfig()
    else:
        sampler = SamplerConfig(
            temperature=config.sampling.temp or 0.6,
            top_p=config.sampling.top_p or 0.95,
            top_k=config.sampling.top_k or 20,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )

    generate = partial(
        generate_mtpk,
        runtime,
        prompt_ids,
        max_tokens=config.max_tokens or 256,
        sampler=sampler,
        speculative_depth=config.speculative_depth or 3,
    )
    return [_run_single(generate) for _ in range(num_runs)]
