import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import mlx_lm
from mlx import nn
from mlx_lm.generate import GenerationResponse
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from common import ChatMessage, get_model_path
from mach import MemoryCounters, get_memory_counters


@dataclass(frozen=True)
class MlxSampling:
    temp: float = 0.0
    top_p: float = 0.0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    top_k: int = 0
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    xtc_special_tokens: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MlxRunRequest:
    # Model location:
    #   str for a Hugging Face repository ID,
    #   Path for a local model directory.
    model_path: str | Path

    # Raw input text or chat messages formatted using the target tokenizer's chat template.
    prompt: str | list[ChatMessage]

    # Maximum number of tokens to generate, excluding prompt tokens.
    max_tokens: int = 256

    # Maximum number of prompt tokens processed per prefill step.
    prefill_step_size: int = 2048

    # Draft model location:
    #   str for a Hugging Face repository ID,
    #   Path for a local model directory.
    #   None disables speculative decoding.
    # Must use the same tokenizer as the target;
    draft_model_path: str | Path | None = None

    # Number of tokens proposed per speculative decoding round; ignored without a draft model.
    draft_tokens: int = 2

    # Sampling parameters
    sampling: MlxSampling | None = None


@dataclass(frozen=True)
class MlxRunResponse:
    text: str

    # Time to the first generated token, in seconds.
    time_first_token: float

    # Prompt processing throughput, in tokens per second.
    prompt_tps: float

    # Token generation throughput, in tokens per second.
    generation_tps: float

    # Average generated tokens per target forward pass: 1 without speculative decoding;
    # may exceed 1 with speculative decoding.
    tokens_per_fp: float

    # Peak allocated memory, in gigabytes (GB).
    peak_memory: float

    # Total prompt processing and generation time, in seconds, excluding model loading.
    duration: float

    # Memory counters collected using the Mach API.
    memory_counters: MemoryCounters


def run(request: MlxRunRequest) -> MlxRunResponse:
    # load main model
    model_path: str = get_model_path(request.model_path)
    model: nn.Module
    tokenizer: TokenizerWrapper
    model, tokenizer = cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(model_path))

    # load draft model
    draft_model: nn.Module | None = None
    if request.draft_model_path is not None:
        draft_model_path: str = get_model_path(request.model_path)
        draft_tokenizer: TokenizerWrapper
        draft_model, draft_tokenizer = cast(tuple[nn.Module, TokenizerWrapper], mlx_lm.load(draft_model_path))
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match target tokenizer")

    # prepare prompt
    prompt: str | list[int]
    if isinstance(request.prompt, str):
        prompt = request.prompt
    else:
        messages = [{"role": message.role.value, "content": message.message} for message in request.prompt]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    # prepare variables
    text: str = ""
    time_first_token: float = -1.0
    draft_flags: list[bool] = []
    response: GenerationResponse | None = None
    memory_graphics_peak_total: int = 0
    memory_with_graphics_peak: MemoryCounters = get_memory_counters()

    # sampling
    sampler: Callable | None = None
    if request.sampling is not None:
        sampler = make_sampler(
            temp=request.sampling.temp,
            top_p=request.sampling.top_p,
            min_p=request.sampling.min_p,
            min_tokens_to_keep=request.sampling.min_tokens_to_keep,
            top_k=request.sampling.top_k,
            xtc_probability=request.sampling.xtc_probability,
            xtc_threshold=request.sampling.xtc_threshold,
            xtc_special_tokens=request.sampling.xtc_special_tokens,
        )

    # create and run inference loop
    time_start: float = time.perf_counter()

    stream: Generator[GenerationResponse] = mlx_lm.stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=request.max_tokens,
        draft_model=draft_model,
        prefill_step_size=request.prefill_step_size,
        num_draft_tokens=request.draft_tokens,
        sampler=sampler,
    )
    for response in stream:
        if time_first_token < 0.0:
            time_first_token = time.perf_counter() - time_start
        text += response.text
        draft_flags.append(response.from_draft)

        memory_curr = get_memory_counters()
        if memory_curr.graphics_total > memory_graphics_peak_total:
            memory_graphics_peak_total = memory_curr.graphics_total
            memory_with_graphics_peak = memory_curr

    time_total: float = time.perf_counter() - time_start

    if response is None:
        raise RuntimeError("Generation did not return a response")

    # Collect tokens per forward passes
    target_forward_passes: int = sum(not flag for flag in draft_flags)
    if draft_flags and draft_flags[-1]:
        # A trailing draft group also used one target verification pass.
        target_forward_passes += 1
    tokens_per_forward_pass: float = response.generation_tokens / target_forward_passes

    return MlxRunResponse(
        text=text,
        time_first_token=time_first_token,
        prompt_tps=response.prompt_tps,
        generation_tps=response.generation_tps,
        tokens_per_fp=tokens_per_forward_pass,
        peak_memory=response.peak_memory,
        duration=time_total,
        memory_counters=memory_with_graphics_peak,
    )
