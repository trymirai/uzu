import json
import math
import secrets
import sys
import time
from collections import deque
from contextlib import redirect_stdout
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path
from types import ModuleType
from typing import Annotated

import typer
from bench import BenchRequest, BenchResponse, BenchSampling
from common import InferenceEngine, run_loop
from mach import get_memory_counters

PROJECT_DIR = Path(__file__).resolve().parent
SPLASH_DIR = PROJECT_DIR / "deps" / "splash"
BUILD_DIR = PROJECT_DIR / "build" / "release"


def load_native() -> ModuleType:
    spec = PathFinder.find_spec("_splash_native", [str(BUILD_DIR / "lib")])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Splash binding is not built; run {PROJECT_DIR / 'bootstrap.sh'} first")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_model(model: str | Path) -> Path:
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
    sampling = sampling or BenchSampling()
    temperature = sampling.temp if sampling.temp is not None else 1.0
    top_p = sampling.top_p if sampling.top_p is not None else 0.95
    top_k = sampling.top_k if sampling.top_k is not None else 20
    # Match Frontend's float32 and native top-k limits for raw prompts too.
    minimum = float.fromhex("0x1p-149")
    if (
        not math.isfinite(temperature)
        or not 0 <= temperature <= 2
        or 0 < temperature < minimum
        or not math.isfinite(top_p)
        or not minimum <= top_p <= 1
        or not 1 <= top_k <= 32
    ):
        raise ValueError("Splash requires temp in [0, 2], top_p in (0, 1], and top_k in [1, 32]")
    if sampling.min_p not in (None, 0.0):
        raise ValueError("Splash does not support nonzero min_p")
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k}


class SplashEngine(InferenceEngine):
    def __init__(self, model: str | Path):
        self.native = load_native()
        self.native.check_device()
        # Upstream is a source distribution, with namespace packages under deps/.
        sys.path.insert(0, str(SPLASH_DIR.parent))

        from splash.server import protocol as wire
        from splash.server.backend import CacheInfo, CallbackStreamer, Job, NativeResult
        from splash.server.chat_templates import ChatTemplates
        from splash.server.constraints import ConstraintFactory, validate_tokenizer
        from splash.server.frontend import Frontend
        from splash.server.metrics import metrics_dict, timings_dict
        from splash.server.thinking import ThinkingCodec
        from transformers import AutoTokenizer

        self._wire = wire
        self._job_type = Job
        self._streamer_type = CallbackStreamer
        self._result_type = NativeResult
        self._cache_type = CacheInfo
        self._metrics = metrics_dict
        self._timings = timings_dict
        self._parser = wire.FrameParser()
        self._events = deque()
        self.closed = False
        self.handle = None
        root = prepare_model(model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            root / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        validate_tokenizer(self.tokenizer)
        chat_templates = ChatTemplates(self.tokenizer)
        try:
            self.handle = self.native.create(str(root), str(BUILD_DIR / "lib" / "splash.metallib"))
            self._step(0.0)
            if len(self._events) != 1 or not isinstance(self._events[0], wire.ReadyEvent):
                raise RuntimeError("Splash did not announce readiness after native warmup")
            ready = self._events.popleft()
            self.frontend = Frontend(
                self.tokenizer,
                None,
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
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        handle, self.handle = self.handle, None
        self._events.clear()
        if handle is not None:
            self.native.close(handle)

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
        # Preserve the common request's precedence when both prompts are present.
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

    def _decode(self, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            step = self._parser.consume(memoryview(data)[offset:])
            if step.issue or not step.consumed_bytes:
                self.close()
                raise RuntimeError(step.issue.describe() if step.issue else "Splash event parser made no progress")
            offset += step.consumed_bytes
            if step.frame is not None:
                self._events.append(self._wire.decode_frame(step.frame))

    def _send(self, frame) -> None:
        # These bytes cross the Python/C++ function boundary in memory.
        self._decode(self.native.receive(self.handle, self._wire.serialize_message(frame)))

    def _step(self, timeout: float) -> None:
        self._decode(self.native.step(self.handle, timeout))

    def _next_event(self, deadline: float):
        while not self._events:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Splash request timed out")
            self._step(min(remaining, 0.1))
        return self._events.popleft()

    def _status(self) -> dict:
        return json.loads(self.native.status(self.handle))

    def _request_frame(self, job):
        wire = self._wire
        remaining = job.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Splash request timed out")
        wall_micros = time.time_ns() // 1000
        remaining_micros = min((1 << 64) - 1 - wall_micros, max(1, int(remaining * 1_000_000)))
        return wire.RequestFrame(
            request_id=job.request_id,
            priority=wire.RequestPriority(job.priority),
            absolute_deadline_unix_micros=wall_micros + remaining_micros,
            remaining_deadline_micros=remaining_micros,
            logical_max_output_tokens=job.max_new_tokens,
            prompt_tokens=tuple(job.prompt_tokens),
            sampling=wire.SamplingParameters(job.temperature, job.top_p, job.top_k),
            seed=job.seed,
            cohort=(
                wire.Cohort.CONSTRAINED
                if job.constraint is not None
                else wire.Cohort.SAMPLING
                if job.temperature > 0
                else wire.Cohort.GREEDY
            ),
            constraint=wire.ConstraintMode.TOKEN_MASK if job.constraint is not None else wire.ConstraintMode.NONE,
            image_spans=job.image_spans,
            image_pixels=job.image_pixels,
            return_progress=job.return_progress,
            score_tokens=job.score_tokens,
        )

    def _run(self, job) -> BenchResponse:
        wire = self._wire
        frame = self._request_frame(job)
        before = self._status()
        memory = get_memory_counters()
        chunks = []
        streamer = self._streamer_type(self.tokenizer, chunks.append)
        cache = self._cache_type()
        started = False
        terminal = False
        completion_tokens = 0
        first_batch_tokens = 0
        try:
            self._send(frame)
            while True:
                event = self._next_event(job.deadline)
                current = get_memory_counters()
                if current.graphics_total > memory.graphics_total:
                    memory = current
                if isinstance(event, wire.ErrorEvent):
                    terminal = True
                    if event.failure_class != wire.FailureClass.REQUEST_ERROR:
                        self.close()
                    raise RuntimeError(event.message.decode("utf-8", errors="replace"))  # noqa: TRY004
                if getattr(event, "request_id", None) != job.request_id:
                    self.close()
                    raise RuntimeError("Splash returned an event for a different request")
                if isinstance(event, wire.StartEvent):
                    if started or not 0 <= event.matched_prompt_tokens <= len(job.prompt_tokens):
                        raise RuntimeError("Splash returned an invalid start event")
                    started = True
                    cache = self._cache_type(
                        status="prefix_hit" if event.cache_disposition == wire.CacheDisposition.PREFIX_HIT else "miss",
                        matched_tokens=event.matched_prompt_tokens,
                        capacity=event.capacity_tokens,
                        slot=event.slot_index,
                    )
                elif isinstance(event, wire.TokensEvent):
                    if not started or event.sequence_offset != completion_tokens:
                        raise RuntimeError("Splash returned an out-of-order token batch")
                    if completion_tokens == 0:
                        first_batch_tokens = len(event.tokens)
                    completion_tokens += len(event.tokens)
                    if completion_tokens > job.max_new_tokens:
                        raise RuntimeError("Splash exceeded max_tokens")
                    if job.constraint is not None:
                        job.constraint.consume(event.tokens)
                    streamer.put_tokens(event.tokens)
                elif isinstance(event, wire.MaskRequestEvent):
                    if job.constraint is None:
                        raise RuntimeError("Splash requested a mask for an unconstrained request")
                    masks = job.constraint.masks(event.simulation_tokens)
                    if len(masks) != event.words_per_mask * event.mask_rows * 4:
                        raise RuntimeError("Splash grammar returned an invalid mask size")
                    self._send(wire.MaskResponseFrame(job.request_id, event.mask_request_id, masks))
                elif isinstance(event, wire.CapacityExhaustedEvent):
                    terminal = True
                    raise RuntimeError("Splash could not allocate the request's KV cache")  # noqa: TRY004
                elif isinstance(event, wire.DoneEvent):
                    terminal = True
                    if event.reason == wire.FinishReason.CANCELLED:
                        raise RuntimeError("Splash request was cancelled")
                    if (
                        not started
                        or event.prompt_tokens != len(job.prompt_tokens)
                        or event.completion_tokens != completion_tokens
                        or (event.reason == wire.FinishReason.LENGTH and completion_tokens != job.max_new_tokens)
                    ):
                        raise RuntimeError("Splash completion counts do not match the request and streamed tokens")
                    streamer.end()
                    result = self._result_type(
                        reason="length" if event.reason == wire.FinishReason.LENGTH else "stop",
                        prompt_tokens=event.prompt_tokens,
                        completion_tokens=event.completion_tokens,
                        start_to_first_token_ms=event.prefill_micros / 1000.0,
                        first_token_to_done_ms=event.decode_micros / 1000.0,
                        request_wall_ms=event.wall_micros / 1000.0,
                        prefill_tokens=max(0, event.prompt_tokens - cache.matched_tokens),
                        cache=cache,
                        first_token_batch_tokens=first_batch_tokens,
                    )
                    break
                elif not isinstance(event, wire.PromptProgressEvent):
                    raise TypeError(f"Unexpected Splash event: {type(event).__name__}")
        except BaseException:
            if not terminal and not self.closed:
                self._cancel_and_drain(job)
            raise
        after = self._status()
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

    def _cancel_and_drain(self, job) -> None:
        # Keep driving the same runtime until the cancelled GPU work retires.
        deadline = time.monotonic() + 155.0
        try:
            self._send(self._wire.CancelFrame(job.request_id))
            while True:
                event = self._next_event(deadline)
                if isinstance(event, self._wire.ErrorEvent):
                    if event.failure_class != self._wire.FailureClass.REQUEST_ERROR:
                        self.close()
                    return
                if getattr(event, "request_id", None) == job.request_id and isinstance(
                    event, (self._wire.DoneEvent, self._wire.CapacityExhaustedEvent)
                ):
                    return
        except BaseException:
            self.close()
            raise


def run(model: Annotated[str, typer.Option("-m", "--model")]) -> None:
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
