"""What the website shows of a release benchmark, from its saved runs.

uv run --project benchman python release_summary.py tag:macs-fleet/0.6.18 summary.json
"""

from pathlib import Path
from typing import Annotated

import typer
from benchman.device import Device
from benchman.report import Run
from benchman.run import load
from benchman.task import summarize
from pydantic import BaseModel, TypeAdapter

SEED = 0
LONDON = "oneoff/london-summary"
LONDON_REPEATS = 3
MIXTURE = "mirai-mixture-slow"
MIXTURE_TASKS = 155


class Result(BaseModel):
    device: Device
    engine: str
    model: str
    source: str
    speculator: str | None
    speculative: bool
    # Over the mixture with speculative decoding, else of the London summary.
    decode_tps: float
    prefill_tps: float
    memory_used: int


def result(run: Run) -> Result:
    name = f"{run.host} {run.engine} {run.model}"
    london = run.responses.get(LONDON)
    if london is None or len(london.responses) != LONDON_REPEATS:
        raise typer.BadParameter(f"{name} has not run {LONDON} {LONDON_REPEATS} times")
    mixture = [task for task in run.responses if task.startswith(f"{MIXTURE}/")] if run.ready.specdec else []
    if run.ready.specdec and len(mixture) != MIXTURE_TASKS:
        raise typer.BadParameter(f"{name} has run {len(mixture)} of the {MIXTURE_TASKS} tasks of {MIXTURE}")
    if unseeded := [task for task in [LONDON, *mixture] if run.tasks[task].seed != SEED]:
        raise typer.BadParameter(f"{name} has not run {unseeded[0]} with seed {SEED}")
    return Result(
        device=run.device,
        engine=run.engine,
        model=run.model,
        source=run.config["model"],
        speculator=run.config.get("draft_model"),
        speculative=run.ready.specdec,
        decode_tps=summarize(run.responses[task] for task in mixture).decode_tps if mixture else london.decode_tps,
        prefill_tps=london.prefill_tps,
        memory_used=london.memory_used,
    )


def main(
    source: Annotated[str, typer.Argument(metavar="FILE|tag:POOL/NAME")],
    target: Annotated[Path, typer.Argument(metavar="FILE")],
) -> None:
    results = [result(run) for run in load(source)]
    target.write_bytes(TypeAdapter(list[Result]).dump_json(results, indent=2))


if __name__ == "__main__":
    typer.run(main)
