import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def serve(model: str, host: str, port: int) -> None:
    from bench_splash import SplashEngine

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    engine = SplashEngine(model)
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        serve(args.model, args.host, args.port)
    except KeyboardInterrupt:
        pass


# The child runs this file in engine-splash's isolated environment. Finish its
# entry point before loading the benchmark client's dependencies and adapter.
if __name__ == "__main__":
    main()
    raise SystemExit(0)
else:
    import httpx2
    from bench import BenchResponse
    from mach import get_memory_counters
    from openai import OpenAI

    from ..util import get_openai_base_url
    from . import ServerEngine


class SplashServerEngine(ServerEngine):
    PROJECT_DIR = Path(__file__).resolve().parents[3] / "engine-splash"

    def create_process(self) -> subprocess.Popen:
        # Reuse the native benchmark's pinned sources and isolated dependencies.
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
        return subprocess.Popen(
            [
                str(self.PROJECT_DIR / ".venv/bin/python"),
                # Keep this splash.py from shadowing the upstream splash package.
                "-P",
                str(Path(__file__).resolve()),
                "--model",
                self.model,
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

    def handle_request(self, request: dict[str, Any]) -> BenchResponse:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Splash server is not running.")

        body = request.copy()
        if body.pop("stream", False):
            raise ValueError("Splash benchmarks require a non-streaming request.")
        model = body.pop("model", self.model)
        messages = body.pop("messages")
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
            started = time.perf_counter()
            response = client.chat.completions.create(model=model, messages=messages, extra_body=body)
            duration = time.perf_counter() - started
            after = self._status(http_client)

        if before["instance"] != after["instance"] or before["transport"]["restarts"] != after["transport"]["restarts"]:
            raise RuntimeError("Splash restarted during the benchmark.")

        usage = response.usage
        extras = response.model_extra or {}
        timings = extras.get("timings")
        latency = (extras.get("metrics") or {}).get("request_latency")
        if usage is None or not isinstance(timings, dict) or not isinstance(latency, dict):
            raise RuntimeError("Splash response is missing usage or timing statistics.")

        # Match engine-splash: these cumulative counters cover native decode
        # batches, including speculative output, and exclude draft forwards.
        batches = after["scheduler"]["decode_batches"] - before["scheduler"]["decode_batches"]
        tokens = after["metrics"]["decode_output_tokens"] - before["metrics"]["decode_output_tokens"]
        if batches < 0 or tokens < 0:
            raise RuntimeError("Splash native counters moved backwards.")

        # The Python HTTP frontend and native inference run in separate
        # processes. Sample both after the response, as for other server engines.
        memory = get_memory_counters(pid=process.pid)
        native_memory = get_memory_counters(pid=after["instance"]["native_pid"])
        return BenchResponse(
            text=response.choices[0].message.content or "",
            tokens_count=usage.completion_tokens,
            time_to_first_token=latency.get("ttft_ms", latency["wall_ms"]) / 1000.0,
            prompt_tps=timings["prompt_per_second"],
            decode_tps=timings["predicted_per_second"],
            tokens_per_forward_pass=tokens / batches if batches else float(usage.completion_tokens > 0),
            duration=duration,
            memory_phys_footprint=memory.phys_footprint + native_memory.phys_footprint,
            memory_resident=memory.resident_size + native_memory.resident_size,
            memory_graphics_total=memory.graphics_total + native_memory.graphics_total,
        )
