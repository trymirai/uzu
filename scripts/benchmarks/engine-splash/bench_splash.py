import json
import math
import queue
import secrets
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated

import typer
from bench import BenchRequest, BenchResponse, BenchSampling
from common import InferenceEngine, run_loop
from mach import get_memory_counters

PROJECT_DIR = Path(__file__).resolve().parent
SPLASH_DIR = PROJECT_DIR / "deps" / "splash"
BUILD_DIR = SPLASH_DIR / "build"
SPLASH_BINARY = BUILD_DIR / "splash"


def get_model_path(model: str | Path) -> Path:
    from splash.install import models, paths, upstream

    path = Path(model).expanduser()
    if path.is_dir():
        root = path.resolve()
    else:
        selection = models.Selection.of(paths.MODELS, str(model))
        upstream.prepare(selection)
        root = selection.link.resolve()

    for name in ("target", "draft", "tokenizer"):
        if not (root / name).is_dir():
            raise ValueError(f"Expected a prepared Splash model directory with {name}/: {root}")

    return root


def sampling_options(sampling: BenchSampling | None) -> dict[str, float | int]:
    # Match Frontend's float32 and native top-k limits for raw prompts too.
    minimum = float.fromhex("0x1p-149")
    sampling = sampling or BenchSampling()

    temperature = sampling.temp if sampling.temp is not None else 1.0
    if not math.isfinite(temperature) or not 0 <= temperature <= 2 or 0 < temperature < minimum:
        raise ValueError("Splash requires temp in [0, 2]")

    top_p = sampling.top_p if sampling.top_p is not None else 0.95
    if not math.isfinite(top_p) or not minimum <= top_p <= 1:
        raise ValueError("Splash requires top_p in (0, 1]")

    top_k = sampling.top_k if sampling.top_k is not None else 20
    if not 1 <= top_k <= 32:
        raise ValueError("Splash requires top_k in [1, 32]")

    if sampling.min_p not in (None, 0.0):
        raise ValueError("Splash does not support nonzero min_p")

    return {"temperature": temperature, "top_p": top_p, "top_k": top_k}


class SplashEngine(InferenceEngine):
    def __init__(self, model: str | Path):
        subprocess.run([str(SPLASH_BINARY), "device-check"], check=True)

        sys.path.insert(0, str(SPLASH_DIR.parent))
        from splash.server import protocol as wire
        from splash.server.backend import Job, NativeBackend
        from splash.server.chat_templates import ChatTemplates
        from splash.server.constraints import ConstraintFactory, validate_tokenizer
        from splash.server.frontend import Frontend
        from splash.server.metrics import metrics_dict, timings_dict
        from splash.server.runtime import MultiplexedRuntime
        from splash.server.thinking import ThinkingCodec
        from transformers import AutoTokenizer

        self._wire = wire
        self._job_type = Job
        self._metrics = metrics_dict
        self._timings = timings_dict
        self.closed = False
        self.process = None
        self.runtime = None
        self.backend = None
        root = get_model_path(model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            root / "tokenizer",
            local_files_only=True,
            trust_remote_code=False,
            fix_mistral_regex=True,
        )
        validate_tokenizer(self.tokenizer)
        chat_templates = ChatTemplates(self.tokenizer)
        self._command = [str(SPLASH_BINARY), "serve-native", str(root / "target"), str(root / "draft"), "auto", "auto"]
        try:
            self.runtime = MultiplexedRuntime(
                self._command,
                process_factory=self._start_process,
                startup_timeout=600.0,
                pending_limit=1,
                eager_start=False,
            )
            self.backend = NativeBackend(self.runtime, self.tokenizer)
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
                str(model),
                ready.max_context_tokens,
                256,
                1800.0,
                ready.max_concurrent_requests,
                constraint_factory=ConstraintFactory(self.tokenizer),
                chat_templates=chat_templates,
                thinking_codec=ThinkingCodec(),
                vision=ready.vision,
            )
            # Fail before accepting requests if child-process accounting is unavailable.
            self._memory()
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
        self.process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        return self.process

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        if self.closed:
            raise RuntimeError("Engine is closed")

        num_runs = request.num_runs if request.num_runs is not None else 1
        if num_runs < 1:
            raise ValueError("num_runs must be 1 or greater")

        if request.speculative_depth not in (None, 7):
            raise ValueError("Splash 1.1.0 uses a fixed speculative_depth of 7; it cannot disable or resize drafting")

        max_tokens = request.max_tokens if request.max_tokens is not None else 256
        if max_tokens < 1:
            raise ValueError("max_tokens must be 1 or greater")

        body = {"max_tokens": max_tokens, **sampling_options(request.sampling)}
        if request.prompt_text is None:
            if request.prompt_chat is None:
                raise ValueError("prompt_text and prompt_chat are None")
            body["messages"] = [message.model_dump(mode="json", exclude_none=True) for message in request.prompt_chat]
            body.update(request.model_dump(mode="json", include={"tools", "tool_choice"}))

        return [self._run(self._prepare(request, body)) for _ in range(num_runs)]

    def _prepare(self, request: BenchRequest, body: dict):
        if request.prompt_text is None:
            job, _, _ = self.frontend.prepare(body)
            return job

        # Splash 1.1.0 exposes raw tokenization but no text-completions route.
        # Submit those tokens to the same native runtime as chat.
        deadline = self.frontend.request_deadline(body)
        tokens = self.frontend.tokenize({"content": request.prompt_text, "add_special": True}, deadline=deadline)
        if not tokens:
            raise ValueError("Input prompt is empty")
        if len(tokens) + body["max_tokens"] > self.frontend.max_context:
            raise ValueError("prompt and max_tokens exceed the context window")

        return self._job_type(
            request_id=next(self.frontend.ids),
            prompt_tokens=tokens,
            max_new_tokens=body["max_tokens"],
            seed=secrets.randbits(64),
            temperature=body["temperature"],
            top_p=body["top_p"],
            top_k=body["top_k"],
            deadline=deadline,
        )

    def _status(self) -> dict:
        # NativeBackend.status() may return cached telemetry on a timeout.
        # A benchmark needs fresh snapshots around its single request.
        event = self.runtime.status()
        snapshot = json.loads(event.json)
        if (
            event.schema_version != self._wire.STATUS_SCHEMA_VERSION
            or not isinstance(snapshot, dict)
            or snapshot.get("schema_version") != self._wire.STATUS_SCHEMA_VERSION
        ):
            raise RuntimeError("Splash status does not match the current schema")
        return snapshot

    def _memory(self):
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Splash process is not running")
        memory = get_memory_counters()
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
        memory = self._memory()
        chunks = []
        submitted = False
        submission_complete = False
        terminal = False

        try:
            submitted = self.backend.submit(job)
            submission_complete = True
            if not submitted:
                raise RuntimeError("Splash request capacity is exhausted")

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

                current = self._memory()
                if current.graphics_total > memory.graphics_total:
                    memory = current
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
            if not submission_complete:
                self.close()
            elif submitted and not terminal:
                self._cancel_and_drain(job, timed_out=isinstance(error, TimeoutError))
            raise

        after = self._status()
        if self.process is not process or self.runtime.restart_count != restarts:
            raise RuntimeError("Splash restarted during the benchmark")

        timings = self._timings(result)
        latency = self._metrics(result)["request_latency"]
        decode_batches = after["scheduler"]["decode_batches"] - before["scheduler"]["decode_batches"]
        decode_tokens = after["metrics"]["decode_output_tokens"] - before["metrics"]["decode_output_tokens"]
        if decode_batches < 0 or decode_tokens < 0:
            raise RuntimeError("Splash native counters moved backwards")

        return BenchResponse(
            text="".join(chunks),
            time_to_first_token=latency.get("ttft_ms", result.request_wall_ms) / 1000.0,
            prompt_tps=timings["prompt_per_second"],
            decode_tps=timings["predicted_per_second"],
            tokens_per_forward_pass=(
                decode_tokens / decode_batches if decode_batches else float(result.completion_tokens > 0)
            ),
            duration=result.request_wall_ms / 1000.0,
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


def run(model: Annotated[str, typer.Option("-m", "--model")]) -> None:
    if not SPLASH_BINARY.is_file():
        subprocess.run([str(PROJECT_DIR / "bootstrap.sh")], check=True, stdout=sys.stderr)

    engine: SplashEngine
    with redirect_stdout(sys.stderr):
        engine = SplashEngine(model)
    try:
        run_loop(engine)
    finally:
        with redirect_stdout(sys.stderr):
            engine.close()


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
