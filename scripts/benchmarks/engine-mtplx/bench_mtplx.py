import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Annotated

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
from mtplx.generation import GenerationOutput, VerifyStrategy, generate_ar, generate_mtpk
from mtplx.profiles import apply_profile_env, runtime_env_overrides_from_contract
from mtplx.runtime import MTPLXRuntime, load
from mtplx.sampling import SamplerConfig


class MTPLXConfig(ModelConfig):
    revision: str | None = None
    speculative_depth: int | None = None


class MTPLXEngine(InferenceEngine):
    runtime: MTPLXRuntime
    speculative_depth: int
    mtp_history_policy: str

    def __init__(self, config: MTPLXConfig):
        self.sampling = get_sampling(config.sampling)
        if self.sampling.min_p not in (None, 0.0):
            raise ValueError("MTPLX does not support min_p sampling")
        super().__init__()
        model_path = get_model_path(config.model, config.revision)
        default_depth = self._configure_profile(model_path)
        self.runtime = load(model_path)
        speculative_depth = config.speculative_depth
        if speculative_depth is None:
            speculative_depth = default_depth if self.runtime.mtp_enabled else 0
        if speculative_depth < 0:
            raise ValueError("speculative_depth must not be negative")
        if speculative_depth and not self.runtime.mtp_enabled:
            raise ValueError("Speculative decoding requires a model with matching MTP weights")
        check_specdec(config, speculative_depth > 0)
        self.speculative_depth = speculative_depth
        self.mtp_history_policy = os.environ.get("MTPLX_MTP_HISTORY_POLICY", "committed")

    def _configure_profile(self, model_path: str) -> int:
        metadata_path = Path(model_path) / "mtplx_runtime.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        config_path = Path(model_path) / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        model_type = config.get("text_config", config).get("model_type")
        # Capture-commit inspects Qwen3-Next-style layers; other architectures
        # (e.g. Qwen Flash-Next) must retain the portable batched verifier.
        capture_verify = model_type in {
            "qwen3_next",
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5_mtp",
        }
        self.verify_strategy: VerifyStrategy = "capture_commit" if capture_verify else "batched"
        self.verify_core = "linear-gdn-from-conv-tape" if capture_verify else "stock"
        runtime_env = runtime_env_overrides_from_contract(metadata)
        if not capture_verify:
            runtime_env["MTPLX_SKIP_VERIFY_SNAPSHOT"] = "0"

        apply_profile_env(
            metadata.get("recommended_profile") or "sustained",
            runtime_env_overrides=runtime_env,
        )

        depth: int | None = metadata.get("recommended_mtp_depth")
        if depth is None:
            depth = metadata.get("mtp_depth_default")

        return int(depth if depth is not None else 3)

    def execute(self, request: BenchRequest) -> BenchResponse:
        depth = self.speculative_depth
        prompt: list[int] = get_tokenized_prompt(request, self.runtime.tokenizer)

        sampling = self.sampling
        sampler = SamplerConfig(
            temperature=1.0 if sampling.temp is None else sampling.temp,
            top_p=1.0 if sampling.top_p is None else sampling.top_p,
            top_k=0 if sampling.top_k is None else sampling.top_k,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )

        tokenizer = self.runtime.tokenizer
        stop_token_ids: set[int] = set(getattr(tokenizer, "eos_token_ids", None) or ())
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)
        if not request.max_output_tokens and not stop_token_ids:
            raise ValueError("Unlimited generation requires an EOS token")

        decoder = (
            partial(
                generate_mtpk,
                speculative_depth=depth,
                mtp_history_policy=self.mtp_history_policy,
                # Pair the profile's snapshot-free path with the CLI's verifier.
                verify_strategy=self.verify_strategy,
                verify_core=self.verify_core,
            )
            if depth
            else generate_ar
        )
        generate = partial(
            decoder,
            self.runtime,
            prompt,
            max_tokens=request.max_output_tokens or sys.maxsize,
            sampler=sampler,
            stop_token_ids=stop_token_ids,
        )
        if request.seed is not None:
            # MTPLX keys speculative draws with the seed's low 31 bits.
            if request.seed >= 2**31:
                raise ValueError("MTPLX supports seeds below 2**31")
            generate = partial(generate, seed=request.seed)
        return self._run(generate, input_tokens_count=len(prompt), speculative=depth > 0)

    def _run(
        self, generate: Callable[..., GenerationOutput], *, input_tokens_count: int, speculative: bool
    ) -> BenchResponse:
        # prepare variables
        time_first_token: float = -1.0
        mem_counters_max: MemoryCounters = get_memory_counters()

        def update_memory(*_progress: object) -> None:
            nonlocal mem_counters_max
            counters = get_memory_counters()
            if counters.graphics_total > mem_counters_max.graphics_total:
                mem_counters_max = counters

        def token_callback(token_ids: list[int]) -> None:
            nonlocal time_first_token
            if token_ids and time_first_token < 0.0:
                time_first_token = time.perf_counter()
            update_memory()

        # create and run inference
        mx.reset_peak_memory()
        energy_start = get_gpu_energy()
        time_start: float = time.perf_counter()
        output = generate(token_callback=token_callback, prefill_callback=update_memory)
        time_end = time.perf_counter()
        energy_gpu = get_gpu_energy() - energy_start
        time_total = time_end - time_start
        update_memory()

        # MTPLX does not invoke token_callback for an immediate EOS.
        if time_first_token < 0.0:
            time_first_token = time_end

        # collect metrics
        decode_tokens = max(0, output.stats.generated_tokens - 1)
        verify_calls = output.stats.verify_calls
        decode_duration = time_end - time_first_token

        return BenchResponse(
            text=output.text,
            finish_reason=get_finish_reason(output.finish_reason),
            input_tokens_count=input_tokens_count,
            output_tokens_count=output.stats.generated_tokens,
            time_to_first_token=(time_first_token - time_start),
            prefill_tps=output.stats.prompt_tps,
            decode_tps=decode_tokens / decode_duration if decode_duration > 0.0 else 0.0,
            tokens_per_forward_pass=decode_tokens / verify_calls if speculative and verify_calls else None,
            duration=time_total,
            energy_gpu=energy_gpu,
            memory_used=mx.get_peak_memory(),
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident=mem_counters_max.resident_size,
            memory_graphics_total=mem_counters_max.graphics_total,
        )


def run(model: Annotated[Path, typer.Argument(help="Engine model configuration JSON.")]) -> None:
    with redirect_stdout(sys.stderr):
        engine = MTPLXEngine(MTPLXConfig.model_validate_json(model.read_text()))
    run_loop(engine)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
