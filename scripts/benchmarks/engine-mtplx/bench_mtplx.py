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
from bench import BenchRequest, BenchResponse
from common import InferenceEngine, get_model_path, get_tokenized_prompt, run_loop
from mach import MemoryCounters, get_memory_counters
from mtplx.generation import GenerationOutput, VerifyStrategy, generate_ar, generate_mtpk
from mtplx.profiles import apply_profile_env, runtime_env_overrides_from_contract
from mtplx.runtime import MTPLXRuntime, load
from mtplx.sampling import SamplerConfig


class MTPLXEngine(InferenceEngine):
    runtime: MTPLXRuntime
    default_speculative_depth: int
    mtp_history_policy: str

    def __init__(self, model: str | Path):
        super().__init__()
        local_path = Path(model).expanduser()
        model_path = str(local_path) if local_path.is_dir() else get_model_path(model)
        default_depth = self._configure_profile(model_path)
        self.runtime = load(model_path)
        self.default_speculative_depth = default_depth if self.runtime.mtp_enabled else 0
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

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        num_runs = request.num_runs if request.num_runs is not None else 1
        if num_runs < 1:
            raise ValueError("num_runs must be 1 or greater")

        if request.max_tokens is not None and request.max_tokens < 0:
            raise ValueError("max_tokens must be 0 or greater")

        depth = request.speculative_depth
        if depth is None:
            depth = self.default_speculative_depth

        if depth < 0:
            raise ValueError("speculative_depth must be 0 or greater")

        if depth and not self.runtime.mtp_enabled:
            raise ValueError("Speculative decoding requires a model with matching MTP weights")

        prompt: list[int] = get_tokenized_prompt(request, self.runtime.tokenizer)

        sampler = SamplerConfig()
        if request.sampling is not None:
            if request.sampling.min_p not in (None, 0.0):
                raise ValueError("MTPLX does not support min_p sampling")
            sampler = SamplerConfig(
                temperature=request.sampling.temp if request.sampling.temp is not None else sampler.temperature,
                top_p=request.sampling.top_p if request.sampling.top_p is not None else sampler.top_p,
                top_k=request.sampling.top_k if request.sampling.top_k is not None else sampler.top_k,
            )

        tokenizer = self.runtime.tokenizer
        stop_token_ids: set[int] = set(getattr(tokenizer, "eos_token_ids", None) or ())
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)
        if not request.max_tokens and not stop_token_ids:
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
            max_tokens=request.max_tokens or sys.maxsize,
            sampler=sampler,
            stop_token_ids=stop_token_ids,
        )
        return [self._run(generate, speculative=depth > 0) for _ in range(num_runs)]

    def _run(self, generate: Callable[..., GenerationOutput], *, speculative: bool) -> BenchResponse:
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
        time_start: float = time.perf_counter()
        output = generate(token_callback=token_callback, prefill_callback=update_memory)
        time_end = time.perf_counter()
        time_total = time_end - time_start
        update_memory()

        # MTPLX does not invoke token_callback for an immediate EOS.
        if time_first_token < 0.0:
            time_first_token = time_end

        # collect metrics
        decode_tokens = max(0, output.stats.generated_tokens - 1)
        if speculative:
            forward_passes = int(output.stats.verify_calls)
            tokens_per_forward_pass = decode_tokens / forward_passes if forward_passes else 0.0
        else:
            # AR does not record verify_calls and emits one token per target forward.
            tokens_per_forward_pass = float(output.stats.generated_tokens > 0)
        decode_duration = time_end - time_first_token

        return BenchResponse(
            text=output.text,
            tokens_count=output.stats.generated_tokens,
            time_to_first_token=(time_first_token - time_start),
            prompt_tps=output.stats.prompt_tps,
            decode_tps=decode_tokens / decode_duration if decode_duration > 0.0 else 0.0,
            tokens_per_forward_pass=tokens_per_forward_pass,
            duration=time_total,
            memory_phys_footprint=mem_counters_max.phys_footprint,
            memory_resident=mem_counters_max.resident_size,
            memory_graphics_total=mem_counters_max.graphics_total,
        )


def run(model: Annotated[str, typer.Option("-m", "--model")]) -> None:
    with redirect_stdout(sys.stderr):
        engine = MTPLXEngine(model)
    run_loop(engine)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
