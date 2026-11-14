import json
import os
import queue
import secrets
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated

import typer
from bench import BenchRequest, BenchResponse, ModelConfig, ReasoningEffort, StochasticSampling
from common import InferenceEngine, check_specdec, get_finish_reason, get_sampling, run_loop
from mach import get_gpu_energy, get_memory_counters

PROJECT_DIR = Path(__file__).resolve().parent
SPLASH_DIR = Path(os.environ.get("BENCHMARK_SPLASH_ROOT", PROJECT_DIR / "deps" / "splash"))
MEMORY_SAMPLE_INTERVAL = 0.1


def get_model_path(model: str, revision: str | None) -> Path:
    from splash.install import models, paths, upstream

    path = Path(model).expanduser()
    if path.is_dir():
        if revision is not None:
            raise ValueError(f"A local model directory has no revision: {model}")
        root = path.resolve()
    else:
        # Benchmark requests contain text only; skip vision weights and warmup.
        selection = models.Selection.of(paths.MODELS, model, revision=revision, language_only=True)
        upstream.prepare(selection)
        root = selection.link.resolve()

    for name in ("target", "draft", "tokenizer"):
        if not (root / name).is_dir():
            raise ValueError(f"Expected a prepared Splash model directory with {name}/: {root}")

    return root


def get_splash_reasoning(effort: ReasoningEffort, reasoning: dict) -> str:
    if reasoning["thinks"] is None or effort != "disabled" and effort not in reasoning["efforts"]:
        raise ValueError(f"The model does not support reasoning_effort {effort}")
    return "none" if effort == "disabled" else effort


def sampling_options(sampling: StochasticSampling) -> dict[str, float | int]:
    if sampling.top_k is None or not 1 <= sampling.top_k <= 32:
        raise ValueError("Splash requires top_k in [1, 32]")

    if sampling.min_p not in (None, 0.0):
        raise ValueError("Splash does not support nonzero min_p")

    return {
        "temperature": 1.0 if sampling.temp is None else sampling.temp,
        "top_p": 1.0 if sampling.top_p is None else sampling.top_p,
        "top_k": sampling.top_k,
    }


class SplashConfig(ModelConfig):
    revision: str | None = None


class SplashEngine(InferenceEngine):
    def __init__(self, config: SplashConfig):
        self.sampling = sampling_options(get_sampling(config.sampling))
        # Every prepared Splash model pairs its target with a draft.
        check_specdec(config, True)
        sys.path.insert(0, str(SPLASH_DIR.parent))
        sys.path.insert(0, str(SPLASH_DIR))
        from splash.install import assembly, models, paths

        self._processes = models.processes
        self._processes.run([str(paths.BINARY), "device-check"], check=True)

        from splash.server import protocol as wire
        from splash.server.backend import NativeBackend
        from splash.server.chat_templates import ChatTemplates
        from splash.server.constraints import ConstraintFactory, validate_tokenizer
        from splash.server.frontend import Frontend
        from splash.server.metrics import metrics_dict, timings_dict
        from splash.server.runtime import MultiplexedRuntime
        from splash.server.thinking import ThinkingCodec
        from transformers import AutoTokenizer

        self._wire = wire
        self._metrics = metrics_dict
        self._timings = timings_dict
        self.closed = False
        self.process = None
        self.runtime = None
        self.backend = None
        root = get_model_path(config.model, config.revision)
        family = assembly.family(root)
        if family is None:
            raise ValueError(f"Splash cannot determine the prepared model's family: {root}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            root / "tokenizer",
            local_files_only=True,
            trust_remote_code=False,
            fix_mistral_regex=True,
        )
        validate_tokenizer(self.tokenizer, family.chat_format)
        chat_templates = ChatTemplates(self.tokenizer)
        self._command = [str(paths.BINARY), "serve-native", str(root), "auto", "auto"]
        try:
            self.runtime = MultiplexedRuntime(
                self._command,
                process_factory=self._start_process,
                startup_timeout=600.0,
                pending_limit=1,
                eager_start=False,
            )
            self.backend = NativeBackend(self.runtime, self.tokenizer, request_logger=lambda record: None)
            # Each run explicitly waits for readiness. Disable the server's
            # background recovery, whose stdout logs would corrupt JSONL output.
            self.runtime.on_engine_failure = None
            if not self.runtime.wait_ready():
                raise RuntimeError("Splash did not become ready after native warmup")

            ready = self.runtime.readiness
            if ready is None:
                raise RuntimeError("Splash did not announce its context window")

            self.frontend = Frontend(
                self.tokenizer,
                self.backend,
                config.model,
                ready.max_context_tokens,
                1800.0,
                ready.max_concurrent_requests,
                constraint_factory=ConstraintFactory(self.tokenizer, family.chat_format),
                chat_templates=chat_templates,
                chat_format=family.chat_format,
                thinking_codec=ThinkingCodec(),
                vision=ready.vision,
            )
            # Fail before accepting requests if child-process accounting is unavailable.
            self._get_memory_counters()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True
        try:
            if self.backend is not None:
                self.backend.close()
            elif self.runtime is not None:
                self.runtime.close()
        finally:
            self.process = None

    def _start_process(self) -> subprocess.Popen:
        self.process = self._processes.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        return self.process

    def execute(self, request: BenchRequest) -> BenchResponse:
        if self.closed:
            raise RuntimeError("Engine is closed")

        body = {
            **self.sampling,
            "messages": [message.model_dump(mode="json", exclude_none=True) for message in request.messages],
            "cache_salt": secrets.token_hex(16),
            **request.model_dump(mode="json", include={"tools", "tool_choice"}),
        }
        if request.max_output_tokens:
            body["max_tokens"] = request.max_output_tokens
        if request.seed is not None:
            body["seed"] = request.seed
        if request.reasoning_effort != "default":
            body["reasoning_effort"] = get_splash_reasoning(request.reasoning_effort, self.frontend.reasoning)
        deadline = self.frontend.request_deadline(body, time.monotonic())
        return self._run(self.frontend.prepare(body, deadline=deadline))

    def _status(self) -> dict:
        # NativeBackend.status() may return cached telemetry on a timeout.
        # A benchmark needs fresh snapshots around its single request.
        event = self.runtime.status()
        snapshot = json.loads(event.json)
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != self._wire.STATUS_SCHEMA_VERSION:
            raise RuntimeError("Splash status does not match the current schema")
        return snapshot

    def _get_memory_counters(self):
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Splash process is not running")
        memory = get_memory_counters()
        if process.pid == os.getpid():
            return memory
        child = get_memory_counters(pid=process.pid)
        # Include Python's tokenizer/frontend as well as the native engine,
        # matching the scope of the other in-process benchmark adapters.
        memory.phys_footprint += child.phys_footprint
        memory.resident_size += child.resident_size
        memory.graphics_total += child.graphics_total
        return memory

    def _run(self, job) -> BenchResponse:
        if not self.runtime.wait_ready():
            raise RuntimeError("Splash runtime is not ready")

        process = self.process
        restarts = self.runtime.restart_count
        before = self._status()
        memory = self._get_memory_counters()
        next_memory_sample = time.monotonic() + MEMORY_SAMPLE_INTERVAL
        chunks = []
        submitted = False
        terminal = False

        energy_start = get_gpu_energy()
        try:
            self.backend.submit(job)
            submitted = True

            while True:
                remaining = job.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Splash request timed out")

                try:
                    kind, value = job.events.get(timeout=min(remaining, 0.1))
                except queue.Empty:
                    kind, value = None, None

                if kind in ("done", "error"):
                    terminal = True
                if kind == "error":
                    raise value
                if self.process is not process or self.runtime.restart_count != restarts:
                    raise RuntimeError("Splash restarted during the benchmark")

                # Detokenization can emit several text fragments per native
                # batch. Keep Mach queries off that per-fragment hot path.
                now = time.monotonic()
                if kind in ("start", "done") or now >= next_memory_sample:
                    current = self._get_memory_counters()
                    if current.graphics_total > memory.graphics_total:
                        memory = current
                    next_memory_sample = now + MEMORY_SAMPLE_INTERVAL
                if kind == "text":
                    chunks.append(value)
                elif kind == "done":
                    result = value
                    if result.reason == "cancelled":
                        raise RuntimeError("Splash request was cancelled")
                    break
                elif kind not in (None, "start", "progress"):
                    raise RuntimeError(f"Unexpected Splash backend event: {kind}")
        except BaseException as error:
            if not submitted:
                self.close()
            elif not terminal:
                self._cancel_and_drain(job, timed_out=isinstance(error, TimeoutError))
            raise

        energy_gpu = get_gpu_energy() - energy_start
        after = self._status()
        if self.process is not process or self.runtime.restart_count != restarts:
            raise RuntimeError("Splash restarted during the benchmark")

        timings = self._timings(result)
        latency = self._metrics(result)["request_latency"]
        decode_batches = after["scheduler"]["decode_batches"] - before["scheduler"]["decode_batches"]
        decode_tokens = after["metrics"]["decode_output_tokens"] - before["metrics"]["decode_output_tokens"]

        # Splash keeps earlier requests' KV pages and states as a prefix cache. Leave its whole KV pool and cached
        # states out of what Metal holds, and count the pages this request's tokens fill.
        kv = after["kv"]
        pages = -(-(result.prompt_tokens + result.completion_tokens) // kv["block_tokens"])
        memory_used = (
            after["memory_actual"]["current_bytes"]
            - kv["allocated_bytes"]
            - after["state"]["bytes"]
            + pages * kv["allocated_bytes"] // kv["pages_allocated"]
        )
        return BenchResponse(
            text="".join(chunks),
            finish_reason=get_finish_reason(result.reason),
            input_tokens_count=result.prompt_tokens,
            output_tokens_count=result.completion_tokens,
            time_to_first_token=latency.get("ttft_ms", result.request_wall_ms) / 1000.0,
            prefill_tps=timings["prompt_per_second"],
            decode_tps=timings["predicted_per_second"],
            tokens_per_forward_pass=decode_tokens / decode_batches if decode_batches else None,
            duration=result.request_wall_ms / 1000.0,
            energy_gpu=energy_gpu,
            memory_used=memory_used,
            memory_phys_footprint=memory.phys_footprint,
            memory_resident=memory.resident_size,
            memory_graphics_total=memory.graphics_total,
        )

    def _cancel_and_drain(self, job, *, timed_out: bool = False) -> None:
        # Let upstream cancellation retire the admitted work before the next
        # request takes its counter baseline. Close if the engine cannot drain.
        try:
            self.backend.cancel(job, timed_out=timed_out)
            deadline = time.monotonic() + 155.0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Splash cancellation timed out")

                kind, _ = job.events.get(timeout=remaining)
                if kind in ("done", "error"):
                    return
        except BaseException:
            self.close()
            raise


def run(model: Annotated[Path, typer.Argument(help="Engine model configuration JSON.")]) -> None:
    if sys.platform != "ios" and "BENCHMARK_SPLASH_ROOT" not in os.environ:
        subprocess.run([str(PROJECT_DIR / "bootstrap.sh")], check=True, stdout=sys.stderr)

    engine: SplashEngine
    with redirect_stdout(sys.stderr):
        engine = SplashEngine(SplashConfig.model_validate_json(model.read_text()))
    try:
        run_loop(engine)
    finally:
        with redirect_stdout(sys.stderr):
            engine.close()


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
