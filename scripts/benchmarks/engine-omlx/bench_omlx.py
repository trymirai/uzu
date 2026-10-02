import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import patch

import typer
from bench import BenchRequest, BenchResponse, BenchSampling
from common import InferenceEngine, get_model_path, get_tokenized_prompt, run_loop
from mach import MemoryCounters, get_memory_counters
from omlx.engine.base import GenerationOutput
from omlx.engine.batched import BatchedEngine
from omlx.engine.dflash import DFlashEngine, is_dflash_compatible
from omlx.model_settings import MAX_LIGHTNING_MTP_DRAFT_TOKENS, ModelSettings
from omlx.utils.model_loading import _checkpoint_has_mtp_weights, _is_mtp_compatible


class OMLXEngine(InferenceEngine):
    engine: BatchedEngine | DFlashEngine
    runner: asyncio.Runner
    closed: bool
    mtp: bool
    mtp_model: Any
    max_depth: int
    default_depth: int

    def __init__(
        self,
        model: str | Path,
        *,
        draft_model: str | Path | None = None,
    ):
        super().__init__()
        self.mtp = False
        self.mtp_model = None
        self.max_depth = 0
        local_path = Path(model).expanduser()
        model_path = str(local_path) if local_path.is_dir() else get_model_path(model)

        if draft_model is not None:
            compatible, reason = is_dflash_compatible(model_path)
            if not compatible:
                raise ValueError(reason)

            draft_path = Path(draft_model).expanduser()
            draft_path = str(draft_path) if draft_path.is_dir() else get_model_path(draft_model)
            config = json.loads((Path(draft_path) / "config.json").read_text())
            self.max_depth = int(config.get("block_size", 0)) - 1
            if self.max_depth < 1:
                raise ValueError("DFlash requires a draft checkpoint with block_size >= 2")

            self.engine = DFlashEngine(
                model_name=model_path,
                draft_model_path=draft_path,
                # Preserve the checkpoint's precision and benchmark full prefills.
                draft_quant_enabled=False,
                model_settings=ModelSettings(
                    dflash_enabled=True,
                    dflash_draft_model=draft_path,
                    dflash_in_memory_cache=False,
                    dflash_verify_mode="dflash",
                ),
            )
        else:
            config = json.loads((Path(model_path) / "config.json").read_text())
            model_type = config.get("model_type")
            self.mtp = (
                _is_mtp_compatible(config, model_type)
                # Gemma 4 MTP requires oMLX's VLM engine, not BatchedEngine.
                and model_type not in ("gemma4", "gemma4_unified")
                and _checkpoint_has_mtp_weights(model_path)
            )
            self.engine = BatchedEngine(
                model_name=model_path,
                model_settings=ModelSettings(mtp_enabled=self.mtp),
            )

        self.default_depth = self.max_depth
        self.runner = asyncio.Runner()
        self.closed = False
        try:
            self.runner.run(self.engine.start())
            if self.mtp:
                from omlx.patches.mlx_lm_mtp.batch_generator import (
                    _model_has_mtp_module,
                    _model_mtp_decode_enabled,
                    _resolve_mtp_chain_depth,
                )

                assert isinstance(self.engine, BatchedEngine)
                self.mtp_model = self.engine._model
                if not _model_has_mtp_module(self.mtp_model) or not _model_mtp_decode_enabled(self.mtp_model):
                    raise ValueError("Detected MTP weights, but oMLX did not load and enable the model's MTP heads")

                chain, self.default_depth, _ = _resolve_mtp_chain_depth(self.mtp_model)
                self.max_depth = MAX_LIGHTNING_MTP_DRAFT_TOKENS if chain else 1
        except BaseException:
            self.close()
            raise

    def _configure_depth(self, depth: int) -> None:
        if depth < 0 or depth > self.max_depth:
            raise ValueError(f"speculative_depth must be between 0 and {self.max_depth} for this engine")

        if isinstance(self.engine, DFlashEngine):
            # DFlash blocks include the target's anchor token as well as drafts.
            self.engine._block_size = depth + 1
        elif self.mtp:
            model = self.mtp_model
            for host in (model, getattr(model, "language_model", model), getattr(model, "_language_model", model)):
                host._omlx_mtp_decode_enabled = depth > 0
                host._omlx_mtp_depth = max(1, depth)

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

        if request.max_tokens is not None and request.max_tokens < 0:
            raise ValueError("max_tokens must be 0 or greater")

        depth = self.default_depth if request.speculative_depth is None else request.speculative_depth
        self._configure_depth(depth)

        prompt: list[int] = get_tokenized_prompt(request, self.engine.tokenizer)

        sampling = request.sampling or BenchSampling()
        generate = partial(
            self.engine.stream_generate,
            prompt=prompt,
            # DFlash compares prompt + output limit against sys.maxsize even
            # with its context cap disabled. Leave room to avoid its AR fallback.
            max_tokens=request.max_tokens or (sys.maxsize - len(prompt) - 1),
            temperature=sampling.temp if sampling.temp is not None else 0.7,
            top_p=sampling.top_p if sampling.top_p is not None else 0.9,
            min_p=sampling.min_p or 0.0,
            top_k=sampling.top_k or 0,
        )

        return [await self._run(generate, len(prompt), depth) for _ in range(num_runs)]

    async def _run(
        self, generate: Callable[[], AsyncIterator[GenerationOutput]], prompt_size: int, depth: int
    ) -> BenchResponse:
        time_first_token: float | None = None
        time_last_token: float | None = None
        completion_tokens = 0
        response: GenerationOutput | None = None
        mem_counters_max: MemoryCounters = get_memory_counters()
        decode_forwards = speculative_forwards = 0
        summary: Any = None
        dflash = isinstance(self.engine, DFlashEngine)

        # Public outputs omit target-forward counts and DFlash token timestamps.
        # Keep hooks and their state local to this run; restore even on failure.
        with ExitStack() as hooks:
            if self.mtp:
                model_type = type(self.mtp_model)
                original_forward = model_type.__call__

                def counted(instance, inputs, *args, **kwargs):
                    nonlocal prompt_size, decode_forwards, speculative_forwards
                    if instance is self.mtp_model:
                        width = inputs.shape[1]
                        if prompt_size > 0:
                            prompt_size -= width
                        else:
                            decode_forwards += 1
                            if kwargs.get("return_hidden") and width > 1:
                                speculative_forwards += 1
                    return original_forward(instance, inputs, *args, **kwargs)

                hooks.enter_context(patch.object(model_type, "__call__", counted))
            elif isinstance(self.engine, DFlashEngine):
                from dflash_mlx.engine.events import SummaryEvent, TokenEvent

                original_events = self.engine._stream_dflash_events

                def measured_events(*args, **kwargs):
                    events, flow, stop_ids = original_events(*args, **kwargs)

                    def measured():
                        nonlocal time_first_token, time_last_token, completion_tokens, summary
                        try:
                            for event in events:
                                if isinstance(event, TokenEvent) and event.token_id not in stop_ids:
                                    time_last_token = time.perf_counter()
                                    if time_first_token is None:
                                        time_first_token = time_last_token
                                    completion_tokens += 1
                                elif isinstance(event, SummaryEvent):
                                    summary = event
                                yield event
                        finally:
                            events.close()

                    return measured(), flow, stop_ids

                hooks.enter_context(patch.object(self.engine, "_stream_dflash_events", measured_events))

            time_start = time.perf_counter()
            async for response in generate():
                if not dflash and response.completion_tokens > completion_tokens:
                    generated_at = response.generated_at
                    if generated_at is None:
                        generated_at = time.perf_counter()
                    if time_first_token is None:
                        time_first_token = generated_at
                    time_last_token = response.generated_until if response.generated_until is not None else generated_at
                    completion_tokens = response.completion_tokens

                mem_counters = get_memory_counters()
                mem_counters_max.phys_footprint = max(mem_counters_max.phys_footprint, mem_counters.phys_footprint)
                mem_counters_max.resident_size = max(mem_counters_max.resident_size, mem_counters.resident_size)
                mem_counters_max.graphics_total = max(mem_counters_max.graphics_total, mem_counters.graphics_total)
        time_total: float = time.perf_counter() - time_start

        if response is None:
            raise RuntimeError("Generation did not return a response")
        if response.finish_reason == "error":
            raise RuntimeError("oMLX generation failed; see engine stderr")

        forward_passes = None
        if self.mtp:
            forward_passes = decode_forwards
            if depth and completion_tokens > 2 and speculative_forwards == 0:
                raise RuntimeError("MTP requested but no speculative verification ran")
            print(
                f"oMLX MTP depth={depth}: {speculative_forwards} speculative verifications, "
                f"{decode_forwards} target decode forwards",
                file=sys.stderr,
            )
        elif dflash:
            if summary is None:
                raise RuntimeError("DFlash did not report a generation summary")
            if summary.fallback_ar:
                raise RuntimeError(f"DFlash fell back to ordinary decoding: {summary.fallback_reason}")
            forward_passes = summary.cycles_completed
            print(
                f"oMLX DFlash: {forward_passes} verification cycles, "
                f"{summary.accepted_from_draft} accepted draft tokens",
                file=sys.stderr,
            )

        # An immediate stop can finish without producing any completion tokens.
        time_to_first_token = time_first_token - time_start if time_first_token is not None else time_total
        decode_time = (
            time_last_token - time_first_token if time_first_token is not None and time_last_token is not None else 0.0
        )
        prompt_tokens = response.prompt_tokens - response.cached_tokens

        return BenchResponse(
            text=response.text,
            tokens_count=completion_tokens,
            time_to_first_token=time_to_first_token,
            prompt_tps=prompt_tokens / time_to_first_token if time_to_first_token > 0.0 else 0.0,
            decode_tps=(completion_tokens - 1) / decode_time if decode_time > 0.0 else 0.0,
            tokens_per_forward_pass=(
                max(0, completion_tokens - 1) / forward_passes
                if forward_passes
                else (1.0 if completion_tokens and forward_passes is None else 0.0)
            ),
            duration=time_total,
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident=mem_counters_max.resident_size,
            memory_graphics_total=mem_counters_max.graphics_total,
        )


def run(
    model: Annotated[str, typer.Option("-m", "--model")],
    draft_model: Annotated[
        str | None, typer.Option("-d", "--draft-model", help="DFlash draft checkpoint path or Hugging Face repository.")
    ] = None,
) -> None:
    engine: OMLXEngine
    with redirect_stdout(sys.stderr):
        engine = OMLXEngine(model, draft_model=draft_model)
    try:
        run_loop(engine)
    finally:
        with redirect_stdout(sys.stderr):
            engine.close()


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
