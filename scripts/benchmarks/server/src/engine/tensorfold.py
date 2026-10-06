import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bench import BenchResponse
from mach import get_memory_counters
from openai import OpenAI

from ..util import get_openai_base_url
from . import ServerEngine


class TensorFoldServerEngine(ServerEngine):
    PROJECT_DIR = Path(__file__).resolve().parents[2] / "tensorfold"

    def create_process(self) -> subprocess.Popen:
        model_path = Path(self.model).expanduser()
        model = str(model_path.resolve()) if model_path.is_dir() else self.model
        subprocess.run(
            ["bash", str(self.PROJECT_DIR / "bootstrap.sh"), model],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
        )
        # Launch the server directly so process.pid identifies the inference
        # process for memory measurements and process-group cleanup.
        return subprocess.Popen(
            [
                str(self.PROJECT_DIR / ".venv/bin/tensorfold"),
                "serve",
                model,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--name",
                self.model,
                "--backend",
                "mlx",
                "--parallel",
                "1",
                "--prompt-cache-gib",
                "0",
                "--snapshot-dir",
                "none",
                "--no-update-check",
            ],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            start_new_session=True,
        )

    def handle_request(self, request: dict[str, Any]) -> BenchResponse:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("TensorFold server is not running.")

        body = request.copy()
        if body.pop("stream", False):
            raise ValueError("TensorFold benchmarks require a non-streaming request.")
        model = body.pop("model", self.model)
        messages = body.pop("messages")
        with OpenAI(base_url=get_openai_base_url(self.host, self.port), api_key="not-needed") as client:
            started = time.perf_counter()
            response = client.chat.completions.create(model=model, messages=messages, extra_body=body)
            duration = time.perf_counter() - started
        memory = get_memory_counters(pid=process.pid)

        usage = response.usage
        extras = response.model_extra or {}
        runtime = extras.get("tensorfold")
        if usage is None or not isinstance(runtime, dict):
            raise RuntimeError("TensorFold response is missing usage or runtime statistics.")

        cached_tokens = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details is not None else 0
        prompt_tokens = max(0, usage.prompt_tokens - (cached_tokens or 0))
        prefill_seconds = runtime.get("prefill_seconds") or 0.0
        speculative = extras.get("speculative") or {}
        # rounds counts decode forwards; each prefill_widths entry is one prompt
        # forward, even when that pass processes several chunks. Exclude drafts.
        forward_passes = (speculative.get("rounds") or 0) + len(runtime.get("prefill_widths") or [])

        return BenchResponse(
            text=response.choices[0].message.content or "",
            tokens_count=usage.completion_tokens,
            time_to_first_token=runtime.get("time_to_first_token") or 0.0,
            prompt_tps=prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0,
            decode_tps=runtime.get("tokens_per_second") or 0.0,
            tokens_per_forward_pass=usage.completion_tokens / forward_passes if forward_passes > 0 else 0.0,
            duration=duration,
            # As for UZU, these are process snapshots after the response.
            memory_phys_footprint=memory.phys_footprint,
            memory_resident=memory.resident_size,
            memory_graphics_total=memory.graphics_total,
        )
