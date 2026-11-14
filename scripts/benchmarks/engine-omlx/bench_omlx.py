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
from omlx.engine.base import GenerationOutput
from omlx.engine.batched import BatchedEngine
from omlx.engine_core import get_mlx_executor
from omlx.model_settings import MAX_LIGHTNING_MTP_DRAFT_TOKENS, ModelSettings
from omlx.speculative.dflash import load_dflash_drafter
from omlx.utils.model_loading import _checkpoint_has_mtp_weights, _is_mtp_compatible


class OMLXConfig(ModelConfig):
    revision: str | None = None
    draft_model: str | None = None
    draft_revision: str | None = None
    speculative_depth: int | None = None


class OMLXEngine(InferenceEngine):
    engine: BatchedEngine
    runner: asyncio.Runner
    closed: bool
    mtp: bool
    mtp_model: Any
    speculative_depth: int

    def __init__(self, config: OMLXConfig):
        super().__init__()
        self.sampling = get_sampling(config.sampling)
        self.mtp = False
        if config.draft_revision is not None and config.draft_model is None:
            raise ValueError("draft_revision requires draft_model")
        self.dflash = config.draft_model is not None
        self.mtp_model = None
        max_depth = 0
        model_path = get_model_path(config.model, config.revision)

        if config.draft_model is not None:
            draft_path = get_model_path(config.draft_model, config.draft_revision)
            draft = json.loads((Path(draft_path) / "config.json").read_text())
            max_depth = int(draft.get("block_size", 0)) - 1
            if max_depth < 1:
                raise ValueError("DFlash requires a draft checkpoint with block_size >= 2")

            self.engine = BatchedEngine(
                model_name=model_path,
                model_settings=ModelSettings(
                    dflash_enabled=True,
                    dflash_draft_model=draft_path,
                    dflash_draft_quant_enabled=False,
                    dflash_verify_mode="dflash",
                ),
            )
        else:
            checkpoint = json.loads((Path(model_path) / "config.json").read_text())
            model_type = checkpoint.get("model_type")
            self.mtp = (
                _is_mtp_compatible(checkpoint, model_type)
                # Gemma 4 MTP requires oMLX's VLM engine, not BatchedEngine.
                and model_type not in ("gemma4", "gemma4_unified")
                and _checkpoint_has_mtp_weights(model_path)
            )
            self.engine = BatchedEngine(
                model_name=model_path,
                model_settings=ModelSettings(mtp_enabled=self.mtp),
            )

        default_depth = max_depth
        self.runner = asyncio.Runner()
        self.closed = False
        try:
            self.runner.run(self.engine.start())
            if self.dflash:
                default_depth = max_depth = self.runner.run(self._load_drafter(draft_path))
                self.mtp_model = self.engine._model
            elif self.mtp:
                from omlx.patches.mlx_lm_mtp.batch_generator import (
                    _model_has_mtp_module,
                    _model_mtp_decode_enabled,
                    _resolve_mtp_chain_depth,
                )

                self.mtp_model = self.engine._model
                if not _model_has_mtp_module(self.mtp_model) or not _model_mtp_decode_enabled(self.mtp_model):
                    raise ValueError("Detected MTP weights, but oMLX did not load and enable the model's MTP heads")

                chain, default_depth, _ = _resolve_mtp_chain_depth(self.mtp_model)
                max_depth = MAX_LIGHTNING_MTP_DRAFT_TOKENS if chain else 1

            self.speculative_depth = default_depth if config.speculative_depth is None else config.speculative_depth
            if not 0 <= self.speculative_depth <= max_depth:
                raise ValueError(f"speculative_depth must be between 0 and {max_depth} for this model")
            check_specdec(config, self.speculative_depth > 0)
            if self.mtp or self.dflash:
                model = self.mtp_model
                for host in (model, getattr(model, "language_model", model), getattr(model, "_language_model", model)):
                    host._omlx_mtp_decode_enabled = self.speculative_depth > 0
                    host._omlx_mtp_depth = max(1, self.speculative_depth)
        except BaseException:
            self.close()
            raise

    async def _load_drafter(self, path: str) -> int:
        def load():
            return load_dflash_drafter(
                path,
                self.engine.prepare_dflash_target(),
                tokenizer=self.engine.tokenizer,
                quant_enabled=False,
                verify_mode="dflash",
            )

        drafter = await asyncio.get_running_loop().run_in_executor(get_mlx_executor(), load)
        self.engine.set_dflash_drafter(drafter)
        return drafter.depth

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True
        try:
            self.runner.run(self.engine.stop())
        finally:
            self.runner.close()

    def execute(self, request: BenchRequest) -> BenchResponse:
        if self.closed:
            raise RuntimeError("Engine is closed")

        return self.runner.run(self._execute(request))

    async def _execute(self, request: BenchRequest) -> BenchResponse:
        prompt: list[int] = get_tokenized_prompt(request, self.engine.tokenizer)

        sampling = self.sampling
        generate = partial(
            self.engine.stream_generate,
            prompt=prompt,
            max_tokens=request.max_output_tokens or (sys.maxsize - len(prompt) - 1),
            temperature=1.0 if sampling.temp is None else sampling.temp,
            top_p=1.0 if sampling.top_p is None else sampling.top_p,
            min_p=0.0 if sampling.min_p is None else sampling.min_p,
            top_k=0 if sampling.top_k is None else sampling.top_k,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=request.seed,
        )

        return await self._run(generate, len(prompt))

    async def _run(self, generate: Callable[[], AsyncIterator[GenerationOutput]], prompt_size: int) -> BenchResponse:
        time_first_token: float | None = None
        time_last_token: float | None = None
        completion_tokens = 0
        response: GenerationOutput | None = None
        mem_counters_max: MemoryCounters = get_memory_counters()
        depth = self.speculative_depth
        speculative = (self.mtp or self.dflash) and depth > 0
        decode_forwards = speculative_forwards = 0

        # Public outputs omit target-forward counts.
        # Keep hooks and their state local to this run; restore even on failure.
        with ExitStack() as hooks:
            if self.mtp or self.dflash:
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
            mx.reset_peak_memory()
            energy_start = get_gpu_energy()
            time_start = time.perf_counter()
            async for response in generate():
                if response.completion_tokens > completion_tokens:
                    generated_at = response.generated_at
                    if generated_at is None:
                        generated_at = time.perf_counter()
                    if time_first_token is None:
                        time_first_token = generated_at
                    time_last_token = response.generated_until if response.generated_until is not None else generated_at
                    completion_tokens = response.completion_tokens

                counters = get_memory_counters()
                if counters.graphics_total > mem_counters_max.graphics_total:
                    mem_counters_max = counters
        time_total: float = time.perf_counter() - time_start
        energy_gpu = get_gpu_energy() - energy_start

        if response is None:
            raise RuntimeError("Generation did not return a response")
        if response.finish_reason == "error":
            raise RuntimeError("oMLX generation failed; see engine stderr")

        if self.mtp or self.dflash:
            if depth and completion_tokens > 2 and speculative_forwards == 0:
                raise RuntimeError("Speculative decoding requested but no verification ran")
            print(
                f"oMLX {'DFlash' if self.dflash else 'MTP'} depth={depth}: "
                f"{speculative_forwards} speculative verifications, "
                f"{decode_forwards} target decode forwards",
                file=sys.stderr,
            )

        # An immediate stop can finish without producing any completion tokens.
        time_to_first_token = time_first_token - time_start if time_first_token is not None else time_total
        decode_time = (
            time_last_token - time_first_token if time_first_token is not None and time_last_token is not None else 0.0
        )
        if response.cached_tokens:
            raise RuntimeError("oMLX reused cached prompt tokens")
        prompt_tokens = response.prompt_tokens
        tokens_per_forward_pass = (completion_tokens - 1) / decode_forwards if speculative and decode_forwards else None

        return BenchResponse(
            text=response.text,
            finish_reason=get_finish_reason(response.finish_reason),
            input_tokens_count=prompt_tokens,
            output_tokens_count=completion_tokens,
            time_to_first_token=time_to_first_token,
            prefill_tps=prompt_tokens / time_to_first_token if time_to_first_token > 0.0 else 0.0,
            decode_tps=(completion_tokens - 1) / decode_time if decode_time > 0.0 else 0.0,
            tokens_per_forward_pass=tokens_per_forward_pass,
            duration=time_total,
            energy_gpu=energy_gpu,
            memory_used=mx.get_peak_memory(),
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident=mem_counters_max.resident_size,
            memory_graphics_total=mem_counters_max.graphics_total,
        )


def run(model: Annotated[Path, typer.Argument(help="Engine model configuration JSON.")]) -> None:
    engine: OMLXEngine
    with redirect_stdout(sys.stderr):
        engine = OMLXEngine(OMLXConfig.model_validate_json(model.read_text()))
    try:
        run_loop(engine)
    finally:
        with redirect_stdout(sys.stderr):
            engine.close()


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
