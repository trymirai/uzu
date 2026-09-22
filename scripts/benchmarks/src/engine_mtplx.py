import time
from dataclasses import dataclass, field
from pathlib import Path

import huggingface_hub
from mtplx.generation import generate_mtpk
from mtplx.runtime import MTPLXRuntime, load
from mtplx.sampling import SamplerConfig

from common import ChatMessage
from mach import MemoryCounters, get_memory_counters


@dataclass(frozen=True)
class MtplxSampling:
    temperature: float | None
    top_p: float | None
    top_k: int | None
    presence_penalty: float | None
    frequency_penalty: float | None


@dataclass(frozen=True)
class MtplxRunRequest:
    # Model location:
    #   str for a Hugging Face repository ID,
    #   Path for a local model directory.
    model: str | Path
    prompt: str | list[ChatMessage]
    max_tokens: int | None = None
    speculative_depth: int | None = None
    sampling: MtplxSampling | None = None


@dataclass(frozen=True)
class MtplxRunResponse:
    text: str = field(repr=False)
    time_to_first_token: float
    prompt_tps: float
    generation_tps: float
    tokens_per_fp: float
    peak_memory: float
    duration: float
    memory_counters: MemoryCounters


def run(request: MtplxRunRequest) -> MtplxRunResponse:
    # load model
    model: str
    if isinstance(request.model, Path):
        model = Path(request.model).expanduser().as_posix()
    else:
        model = huggingface_hub.snapshot_download(
            repo_id=request.model,
            cache_dir=Path("~/.cache/huggingface/hub").expanduser(),
            local_files_only=False,
        )
    runtime: MTPLXRuntime = load(model)

    # prepare prompt
    prompt_ids: list[int]
    if isinstance(request.prompt, str):
        prompt_ids = runtime.tokenizer.encode(request.prompt)
    else:
        messages = [{"role": message.role.value, "content": message.message} for message in request.prompt]
        prompt_ids = runtime.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    # prepare variables
    time_first_token: float = -1.0
    mem_graphics_max: int = 0
    mem_counters_max: MemoryCounters = get_memory_counters()

    sampler: SamplerConfig
    if request.sampling is None:
        sampler = SamplerConfig()
    else:
        sampler = SamplerConfig(
            temperature=request.sampling.temperature or 0.6,
            top_p=request.sampling.top_p or 0.95,
            top_k=request.sampling.top_k or 20,
            presence_penalty=request.sampling.presence_penalty or 0.0,
            frequency_penalty=request.sampling.frequency_penalty or 0.0,
        )

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
    output = generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=request.max_tokens or 256,
        sampler=sampler,
        speculative_depth=request.speculative_depth or 3,
        token_callback=token_callback,
    )

    # collect metrics
    forward_passes = int(output.stats.verify_calls)
    if forward_passes == 0:
        raise RuntimeError("Forward passes is 0")
    tokens_per_forward_pass = max(0, output.stats.generated_tokens - 1) / forward_passes

    return MtplxRunResponse(
        text=output.text,
        time_to_first_token=(time_first_token - time_start),
        prompt_tps=output.stats.prompt_tps,
        generation_tps=output.stats.decode_tok_s,
        tokens_per_fp=tokens_per_forward_pass,
        peak_memory=output.stats.peak_memory_bytes,
        duration=output.stats.elapsed_s,
        memory_counters=mem_counters_max,
    )
