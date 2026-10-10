import re
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, TextIO

from bench import BenchResponse
from mach import get_memory_counters
from openai import OpenAI

from ..util import get_openai_base_url
from . import ServerEngine

# UZU exposes these metrics in its completion log, not the HTTP response.
COMPLETION_STATS = re.compile(
    r"^\[(?P<request_id>[0-9a-f]{8})\] completed in [\d.]+s, finish=\S+, "
    r"prompt \d+ tok \(\d+ cached, \d+ prefilled @ (?P<prompt_tps>[\d.]+|-) tok/s\)"
    r"(?:, ttft (?P<time_to_first_token>[\d.]+)s)?"
    r"(?:, decode \d+ tok @ (?P<decode_tps>[\d.]+|-) tok/s)?"
    r"(?:, spec (?P<tokens_per_forward_pass>[\d.]+) tok/pass \(\d+ passes\))?"
)


class UzuServerEngine(ServerEngine):
    REPO_ROOT = Path(__file__).resolve().parents[5]

    def __init__(self, host: str, port: int, model: str):
        super().__init__(host, port, model)
        self._stats: Queue[tuple[str, dict[str, float]] | None] = Queue()
        self._log_thread: Thread | None = None

    def create_process(self) -> subprocess.Popen:
        process = subprocess.Popen(
            [
                "cargo",
                "run",
                "--release",
                "-p",
                "cli",
                "--",
                "server",
                "--no-prefix-cache",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--model",
                self.model,
            ],
            cwd=UzuServerEngine.REPO_ROOT,
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

    def handle_request(self, request: dict[str, Any]) -> BenchResponse:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("UZU server is not running.")

        body = request.copy()
        model = body.pop("model", self.model)
        messages = body.pop("messages")
        with OpenAI(
            base_url=get_openai_base_url(self.host, self.port),
            api_key="not-needed",
        ) as client:
            started = time.perf_counter()
            response = client.chat.completions.create(model=model, messages=messages, extra_body=body)
            duration = time.perf_counter() - started

        memory = get_memory_counters(pid=process.pid)
        usage = response.usage
        tokens_count = usage.completion_tokens if usage is not None else 0
        stats = self._response_stats(response.id)

        return BenchResponse(
            text=response.choices[0].message.content or "",
            tokens_count=tokens_count,
            time_to_first_token=stats["time_to_first_token"],
            prompt_tps=stats["prompt_tps"],
            decode_tps=stats["decode_tps"],
            tokens_per_forward_pass=stats["tokens_per_forward_pass"],
            duration=duration,
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
                        fields = match.groupdict()
                        request_id = fields.pop("request_id")
                        stats = {
                            name: float(value) if value not in (None, "-") else 0.0 for name, value in fields.items()
                        }
                        self._stats.put((request_id, stats))
                    sys.stderr.write(line)
                    sys.stderr.flush()
        finally:
            self._stats.put(None)

    def _response_stats(self, response_id: str) -> dict[str, float]:
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
