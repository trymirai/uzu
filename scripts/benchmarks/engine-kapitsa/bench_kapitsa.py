import ctypes
import json
import os
import sys
import time
from contextlib import redirect_stdout
from ctypes import CFUNCTYPE, POINTER, byref, c_char_p, c_int, c_size_t, c_void_p
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import typer
from bench import BenchRequest, BenchResponse, ModelConfig, ReadyResponse, ReasoningEffort
from common import (
    InferenceEngine,
    check_specdec,
    get_finish_reason,
    get_model_path,
    get_sampling,
    get_supported_reasoning_efforts,
    get_template_reasoning,
    run_loop,
)
from mach import get_gpu_energy, get_memory_counters

PROJECT_DIR = Path(__file__).resolve().parent
DELTA = CFUNCTYPE(c_int, c_char_p, c_size_t, c_void_p)
# The thinking level Kapitsa reports for each effort it has. It reads `high` as `xhigh`.
THINKING_LEVELS: dict[ReasoningEffort, str] = {"disabled": "off", "low": "low", "medium": "medium", "xhigh": "xhigh"}


class KapitsaConfig(ModelConfig):
    revision: str | None = None
    draft_model: str | None = None
    draft_revision: str | None = None


class KapitsaEngine(InferenceEngine):
    """Kapitsa in this process, through its SDK's C interface."""

    def __init__(self, config: KapitsaConfig):
        self.sampling = sampling = get_sampling(config.sampling)
        if sampling is not None and sampling.min_p not in (None, 0):
            raise ValueError("Kapitsa does not support min_p sampling")
        # Kapitsa clamps top_k to this range.
        if sampling is not None and sampling.temp != 0 and (sampling.top_k is None or not 1 <= sampling.top_k <= 32):
            raise ValueError("Kapitsa requires top_k in [1, 32]")
        if config.draft_revision is not None and config.draft_model is None:
            raise ValueError("draft_revision requires draft_model")
        check_specdec(config, config.draft_model is not None)
        self.config = config

        library = os.environ.get("BENCHMARK_KAPITSA_LIBRARY")
        if library is None:
            library = str(PROJECT_DIR / "deps/kapitsa/build/libkapitsa_sdk.dylib")
        self.library = ctypes.CDLL(library)
        self.library.kapitsa_engine_create.argtypes = [c_char_p, POINTER(c_void_p), POINTER(c_void_p)]
        self.library.kapitsa_request_create.argtypes = [c_void_p, c_char_p, POINTER(c_void_p), POINTER(c_void_p)]
        self.library.kapitsa_generate.argtypes = [c_void_p, DELTA, c_void_p, POINTER(c_void_p), POINTER(c_void_p)]
        self.library.kapitsa_request_destroy.argtypes = [c_void_p]
        self.library.kapitsa_engine_destroy.argtypes = [c_void_p]
        self.library.kapitsa_string_free.argtypes = [c_void_p]

        options: dict[str, Any] = {
            "model_dir": get_model_path(config.model, config.revision),
            "slots": 1,
            "disk_cache": "0",
        }
        if config.draft_model is not None:
            options["dflash"] = get_model_path(config.draft_model, config.draft_revision)
        else:
            options |= {"dflash": "none", "spec_depth": 0}
        self.engine = c_void_p()
        error = c_void_p()
        status = self.library.kapitsa_engine_create(json.dumps(options).encode(), byref(self.engine), byref(error))
        self._check(status, error)

    def close(self) -> None:
        self.library.kapitsa_engine_destroy(self.engine)

    def _check(self, status: int, error: c_void_p) -> None:
        message = ctypes.string_at(error).decode() if error else ""
        self.library.kapitsa_string_free(error)
        if status:
            raise RuntimeError(f"Kapitsa failed ({status}): {message}")

    def _generate(self, request: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """The result of a request and its time to first token."""
        handle = c_void_p()
        error = c_void_p()
        encoded = json.dumps(request).encode()
        self._check(self.library.kapitsa_request_create(self.engine, encoded, byref(handle), byref(error)), error)
        first = None

        @DELTA
        def delta(data: bytes, _length: int, _user: object) -> int:
            nonlocal first
            if first is None and any(json.loads(data).values()):
                first = time.perf_counter()
            return 1

        result = c_void_p()
        started = time.perf_counter()
        status = self.library.kapitsa_generate(handle, delta, None, byref(result), byref(error))
        self.library.kapitsa_request_destroy(handle)
        output = json.loads(ctypes.string_at(result)) if result else {}
        self.library.kapitsa_string_free(result)
        self._check(status, error)
        if first is None:
            raise RuntimeError("Kapitsa generated nothing")
        return output, first - started

    def _honors(self, effort: ReasoningEffort) -> bool:
        """Kapitsa's slot caches the last conversation, with no switch for it. This other conversation evicts it, and
        reports the thinking level Kapitsa makes of the effort."""
        reasoning = get_template_reasoning(effort)
        other, _ = self._generate(
            {"messages": [{"role": "user", "content": "."}], "max_tokens": 1, "template_kwargs": reasoning}
        )
        return other["thinking_level"] == THINKING_LEVELS.get(effort)

    def ready(self) -> ReadyResponse:
        return ReadyResponse(
            specdec=self.config.draft_model is not None,
            supported_reasoning_efforts=get_supported_reasoning_efforts(self._honors),
        )

    def execute(self, request: BenchRequest) -> BenchResponse:
        if request.tool_choice is not None:
            raise ValueError("Kapitsa's SDK does not support tool_choice")
        if request.seed == 2**32 - 1:
            raise ValueError("Kapitsa has no seed after 2**32 - 1")
        effort = request.reasoning_effort
        if not self._honors(effort) and effort != "default":
            raise ValueError(f"The model does not support reasoning_effort {effort}")

        sampling = self.sampling
        body = {
            "messages": [message.model_dump(mode="json", exclude_none=True) for message in request.messages],
            # Without a limit, the context ends the reply.
            "max_tokens": request.max_output_tokens or 2**20,
            "template_kwargs": get_template_reasoning(effort),
            **request.model_dump(mode="json", include={"tools"}, exclude_none=True),
        }
        if sampling is not None:
            body |= {
                "temperature": 1.0 if sampling.temp is None else sampling.temp,
                "top_k": 0 if sampling.top_k is None else sampling.top_k,
                "top_p": 1.0 if sampling.top_p is None else sampling.top_p,
            }
        if request.seed is not None:
            # 0 is Kapitsa's random seed.
            body["seed"] = request.seed + 1
        energy_start = get_gpu_energy()
        started = time.perf_counter()
        result, time_to_first_token = self._generate(body)
        duration = time.perf_counter() - started
        energy_gpu = get_gpu_energy() - energy_start
        memory = get_memory_counters()

        if result["cached_tokens"]:
            raise RuntimeError("Kapitsa reused cached prompt tokens")
        decode_tokens = result["completion_tokens"] - 1
        decode_seconds = result["decode_ms"] / 1000
        # The first token takes no decoding cycle.
        speculative = decode_tokens > 0 and result["kflash_source"] != "none"
        if decode_tokens > 0:
            check_specdec(self.config, speculative)

        return BenchResponse(
            text=result["reasoning"] + result["content"],
            finish_reason=get_finish_reason(result["finish_reason"]),
            input_tokens_count=result["prompt_tokens"],
            output_tokens_count=result["completion_tokens"],
            time_to_first_token=time_to_first_token,
            prefill_tps=result["prompt_tokens"] * 1000 / result["prefill_ms"],
            decode_tps=decode_tokens / decode_seconds if decode_seconds > 0 else 0.0,
            tokens_per_forward_pass=decode_tokens / result["kflash_cycles"] if speculative else None,
            forwards_per_second=result["kflash_cycles"] / decode_seconds if speculative else None,
            duration=duration,
            energy_gpu=energy_gpu,
            # Kapitsa reports no memory. Its Metal allocation keeps the KV room of the longest request so far.
            memory_used=memory.graphics_total,
            memory_phys_footprint=memory.phys_footprint,
            memory_resident=memory.resident_size,
            memory_graphics_total=memory.graphics_total,
        )


def main() -> None:
    typer.run(run)


def run(model: Annotated[Path, typer.Argument(help="Kapitsa model configuration JSON.")]) -> None:
    config = KapitsaConfig.model_validate_json(model.read_text())
    # Kapitsa's home keeps what its drafting learns, which must not reach the next run.
    with TemporaryDirectory(prefix="kapitsa-home-") as home:
        # Initialize the Neural Engine before the first request.
        os.environ |= {"KAPITSA_HOME": home, "NI_ANE_ASYNC": "0"}
        with redirect_stdout(sys.stderr):
            engine = KapitsaEngine(config)
        try:
            run_loop(engine)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
