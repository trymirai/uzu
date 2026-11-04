import argparse
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def serve(config: str, host: str, port: int) -> None:
    from bench_splash import SplashConfig, SplashEngine

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    engine = SplashEngine(SplashConfig.model_validate_json(config))
    try:
        # SplashEngine initializes imports from the pinned engine-splash/deps.
        from splash.server.server import FrontendServer

        class BenchmarkServer(FrontendServer):
            def status(self):
                status = super().status()
                process = engine.process
                status["instance"]["native_pid"] = (
                    process.pid if process is not None and process.poll() is None else None
                )
                return status

        with BenchmarkServer((host, port), engine.frontend, request_capacity=1, webui=False) as server:
            server.serve_forever()
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Splash benchmark engine over HTTP.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        serve(args.config, args.host, args.port)
    except KeyboardInterrupt:
        pass


# The child runs this file in engine-splash's isolated environment. Finish its
# entry point before loading the benchmark client's dependencies and adapter.
if __name__ == "__main__":
    main()
    raise SystemExit(0)
else:
    import httpx2
    from bench import BenchResponse, ModelConfig, ReasoningEffort
    from common import get_finish_reason
    from mach import get_gpu_energy, get_memory_counters
    from openai import OpenAI

    from ..util import get_openai_base_url
    from . import ServerEngine, get_openai_reasoning


class SplashConfig(ModelConfig):
    revision: str | None = None


class SplashServerEngine(ServerEngine):
    supports_min_p = False
    PROJECT_DIR = Path(__file__).resolve().parents[3] / "engine-splash"

    def __init__(self, host: str, port: int, config: SplashConfig):
        super().__init__(host, port, config)
        self.config = config

    def create_process(self) -> subprocess.Popen:
        # Reuse the native benchmark's pinned sources and isolated dependencies.
        python = os.environ.get("BENCHMARK_SPLASH_PYTHON")
        if python is None:
            environment = os.environ.copy()
            environment.pop("VIRTUAL_ENV", None)
            subprocess.run(
                ["uv", "sync", "--project", str(self.PROJECT_DIR), "--frozen"],
                check=True,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
            )
            subprocess.run(
                ["bash", str(self.PROJECT_DIR / "bootstrap.sh")],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
            )
            python = str(self.PROJECT_DIR / ".venv/bin/python")
        return subprocess.Popen(
            [
                python,
                # Keep this splash.py from shadowing the upstream splash package.
                "-P",
                str(Path(__file__).resolve()),
                "--config",
                self.config.model_dump_json(),
                "--host",
                self.host,
                "--port",
                str(self.port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            start_new_session=True,
        )

    def _status(self, client: httpx2.Client) -> dict[str, Any]:
        response = client.get(get_openai_base_url(self.host, self.port).copy_with(path="/status"), timeout=30)
        response.raise_for_status()
        status = response.json()
        transport = status.get("transport", {})
        if not status.get("ready") or not transport.get("ready") or transport.get("status_stale", True):
            raise RuntimeError("Splash did not return fresh, ready native statistics.")
        instance = status.get("instance", {})
        if self.process is None or instance.get("pid") != self.process.pid:
            raise RuntimeError("Splash status belongs to a different server process.")
        if not isinstance(instance.get("native_pid"), int) or instance["native_pid"] <= 0:
            raise RuntimeError("Splash status is missing its native process ID.")
        return status

    def get_reasoning(self, effort: ReasoningEffort, messages: list[Any], body: dict[str, Any]) -> dict[str, Any]:
        response = httpx2.get(f"{get_openai_base_url(self.host, self.port)}/models", timeout=30)
        response.raise_for_status()
        reasoning = response.json()["data"][0]["reasoning"]
        if reasoning["thinks"] is None or effort != "disabled" and effort not in reasoning["efforts"]:
            raise ValueError(f"The model does not support reasoning_effort {effort}")
        return get_openai_reasoning(effort)

    def handle_request(self, messages: list[Any], body: dict[str, Any]) -> BenchResponse:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Splash server is not running.")

        body["cache_salt"] = secrets.token_hex(16)
        with (
            httpx2.Client(timeout=1800) as http_client,
            OpenAI(
                base_url=get_openai_base_url(self.host, self.port),
                api_key="not-needed",
                http_client=http_client,
                max_retries=0,
            ) as client,
        ):
            before = self._status(http_client)
            energy_start = get_gpu_energy()
            started = time.perf_counter()
            response = client.chat.completions.create(model=self.model, messages=messages, extra_body=body)
            duration = time.perf_counter() - started
            energy_gpu = get_gpu_energy() - energy_start
            after = self._status(http_client)

        if before["instance"] != after["instance"] or before["transport"]["restarts"] != after["transport"]["restarts"]:
            raise RuntimeError("Splash restarted during the benchmark.")

        usage = response.usage
        extras = response.model_extra or {}
        timings = extras.get("timings")
        latency = (extras.get("metrics") or {}).get("request_latency")
        if usage is None or not isinstance(timings, dict) or not isinstance(latency, dict):
            raise RuntimeError("Splash response is missing usage or timing statistics.")

        decode_batches = after["scheduler"]["decode_batches"] - before["scheduler"]["decode_batches"]
        decode_tokens = after["metrics"]["decode_output_tokens"] - before["metrics"]["decode_output_tokens"]

        # The Python HTTP frontend and native inference run in separate
        # processes. Sample both after the response, as for other server engines.
        memory = get_memory_counters(pid=process.pid)
        native_memory = get_memory_counters(pid=after["instance"]["native_pid"])
        # Splash keeps earlier requests' KV pages and states as a prefix cache. Leave its whole KV pool and cached
        # states out of what Metal holds, and count the pages this request's tokens fill.
        kv = after["kv"]
        pages = -(-(usage.prompt_tokens + usage.completion_tokens) // kv["block_tokens"])
        memory_used = (
            after["memory_actual"]["current_bytes"]
            - kv["allocated_bytes"]
            - after["state"]["bytes"]
            + pages * kv["allocated_bytes"] // kv["pages_allocated"]
        )
        return BenchResponse(
            text=response.choices[0].message.content or "",
            finish_reason=get_finish_reason(response.choices[0].finish_reason),
            input_tokens_count=usage.prompt_tokens,
            output_tokens_count=usage.completion_tokens,
            time_to_first_token=latency.get("ttft_ms", latency["wall_ms"]) / 1000.0,
            prefill_tps=timings["prompt_per_second"],
            decode_tps=timings["predicted_per_second"],
            tokens_per_forward_pass=decode_tokens / decode_batches if decode_batches else None,
            duration=duration,
            energy_gpu=energy_gpu,
            memory_used=memory_used,
            memory_phys_footprint=memory.phys_footprint + native_memory.phys_footprint,
            memory_resident=memory.resident_size + native_memory.resident_size,
            memory_graphics_total=memory.graphics_total + native_memory.graphics_total,
        )
