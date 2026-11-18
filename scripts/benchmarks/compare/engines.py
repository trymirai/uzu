"""Run each compared engine over the shared JSON Lines transport."""

from pathlib import Path

import engine_process
import typer
from bench import BenchRequest, BenchResponse
from server.util import await_cooldown

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "workspace/compare-logs"


def run_engine(
    name: str, command: list[str], requests: list[BenchRequest], wait_cooldown: bool, timeout: float
) -> list[list[BenchResponse]]:
    """Each request's responses from one engine process. A warmup run of the first request finishes build, model
    loading and first-use compilation (vLLM's Triton kernels autotune on their first decode) before the optional
    cooldown, which also precedes every measured request."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    typer.echo(f"{name}: building, loading and warming up (log: {log})", err=True)
    with engine_process.run_engine(command, ROOT, log, timeout) as engine:
        engine.request(requests[0].model_copy(update={"num_runs": 1}))
        results = []
        for index, request in enumerate(requests):
            if wait_cooldown:
                await_cooldown()
            typer.echo(f"{name}: request {index + 1}/{len(requests)}, {request.num_runs} runs", err=True)
            results.append(engine.request(request))
        return results


def prompt_tokens(response: BenchResponse) -> int:
    return round(response.prompt_tps * response.time_to_first_token)
