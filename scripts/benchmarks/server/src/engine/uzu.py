import os
import re
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, TextIO

from bench import BenchResponse, ModelConfig, ReasoningEffort
from common import check_specdec, get_finish_reason
from mach import get_gpu_energy, get_memory_counters
from openai import OpenAI

from ..util import get_openai_base_url
from . import ServerEngine

# UZU exposes these metrics in its completion log, not the HTTP response.
COMPLETION_STATS = re.compile(
    r"^\[(?P<request_id>[0-9a-f]{8})\] completed in [\d.]+s, finish=\S+, "
    r"prompt \d+ tok \((?P<cached>\d+) cached, \d+ prefilled @ (?P<prefill_tps>[\d.]+|-) tok/s\)"
    r"(?:, ttft (?P<time_to_first_token>[\d.]+)s)?"
    r"(?:, decode \d+ tok @ (?P<decode_tps>[\d.]+|-) tok/s)?"
    r"(?:, spec (?P<tokens_per_forward_pass>[\d.]+) tok/pass \(\d+ passes\))?"
)


def number(value: str | None) -> float:
    return float(value) if value not in (None, "-") else 0.0


class UzuServerEngine(ServerEngine):
    supports_min_p = False
    REPO_ROOT = Path(__file__).resolve().parents[5]

    def __init__(self, host: str, port: int, config: ModelConfig):
        super().__init__(host, port, config)
        self.config = config
        self._stats: Queue[tuple[str, dict[str, str | None]] | None] = Queue()
        self._log_thread: Thread | None = None

    def create_process(self) -> subprocess.Popen:
        binary = os.environ.get("BENCHMARK_UZU_BINARY")
        command = [binary] if binary else ["cargo", "run", "--release", "-p", "cli", "--"]
        process = subprocess.Popen(
            command
            + [
                "server",
                "--no-prefix-cache",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--model",
                self.model,
            ],
            cwd=None if binary else UzuServerEngine.REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert process.stdout is not None
        self._stats = Queue()
        self._log_thread = Thread(target=self._read_logs, args=(process.stdout,), daemon=True)
        self._log_thread.start()
        return process

    def get_reasoning(self, effort: ReasoningEffort, messages: list[Any], body: dict[str, Any]) -> dict[str, Any]:
        return {"reasoning_effort": effort}

    def handle_request(self, messages: list[Any], body: dict[str, Any]) -> BenchResponse:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("UZU server is not running.")

        with OpenAI(
            base_url=get_openai_base_url(self.host, self.port),
            api_key="not-needed",
        ) as client:
            energy_start = get_gpu_energy()
            started = time.perf_counter()
            response = client.chat.completions.create(model=self.model, messages=messages, extra_body=body)
            duration = time.perf_counter() - started
            energy_gpu = get_gpu_energy() - energy_start

        memory = get_memory_counters(pid=process.pid)
        usage = response.usage
        if usage is None:
            raise RuntimeError("UZU response is missing usage statistics.")
        stats = self._response_stats(response.id)
        if stats["cached"] != "0":
            raise RuntimeError("UZU reused cached prompt tokens")
        # The server reveals speculation only in a reply's statistics, and the first token needs no decode pass.
        if usage.completion_tokens > 1:
            check_specdec(self.config, stats["tokens_per_forward_pass"] is not None)

        return BenchResponse(
            text=response.choices[0].message.content or "",
            finish_reason=get_finish_reason(response.choices[0].finish_reason),
            input_tokens_count=usage.prompt_tokens,
            output_tokens_count=usage.completion_tokens,
            time_to_first_token=number(stats["time_to_first_token"]),
            prefill_tps=number(stats["prefill_tps"]),
            decode_tps=number(stats["decode_tps"]),
            tokens_per_forward_pass=None if (spec := stats["tokens_per_forward_pass"]) is None else float(spec),
            duration=duration,
            energy_gpu=energy_gpu,
            # The server reports no memory; the kernel's count of its graphics memory is its Metal allocation.
            memory_used=memory.graphics_total,
            memory_phys_footprint=memory.phys_footprint,
            memory_resident=memory.resident_size,
            memory_graphics_total=memory.graphics_total,
        )

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            if self._log_thread is not None:
                self._log_thread.join(timeout=5)
                self._log_thread = None

    def _read_logs(self, output: TextIO) -> None:
        try:
            with output:
                for line in output:
                    match = COMPLETION_STATS.match(line)
                    if match is not None:
                        stats = match.groupdict()
                        self._stats.put((stats.pop("request_id"), stats))
                    sys.stderr.write(line)
                    sys.stderr.flush()
        finally:
            self._stats.put(None)

    def _response_stats(self, response_id: str) -> dict[str, str | None]:
        while True:
            try:
                entry = self._stats.get(timeout=5)
            except Empty as error:
                raise RuntimeError(f"UZU server did not log statistics for response {response_id}.") from error
            if entry is None:
                raise RuntimeError("UZU server output closed before response statistics were received.")
            request_id, stats = entry
            if request_id == response_id[:8]:
                return stats
