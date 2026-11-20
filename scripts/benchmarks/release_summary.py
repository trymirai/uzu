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

LONDON = "oneoff/london-summary"
LONDON_WARMUPS, LONDON_REPEATS = 1, 3
# The mixtures, with how many tasks the whole of each is.
MIXTURES = {"mirai-mixture-fast": 21, "mirai-mixture-slow": 155}


class Measured(BaseModel):
    speculative: bool
    # The mean over a mixture with speculative decoding, else of the London summary; and how many times that ran.
    decode_dataset: str
    decode_repeats: int
    decode_tps: float
    # Of the same, and only with speculative decoding.
    tokens_per_forward_pass: float | None
    forwards_per_second: float | None
    # Of the London summary.
    prefill_tps: float
    time_to_first_token: float
    memory_used: int


class Result(BaseModel):
    device: Device
    engine: str
    model: str
    source: str
    speculator: str | None
    # One of the two: a failure is how a run ended, with no numbers to show.
    measured: Measured | None
    failure: str | None


def measured(run: Run) -> Measured:
    name = f"{run.host} {run.engine} {run.model}"
    london = run.responses.get(LONDON)
    if london is None or (run.tasks[LONDON].warmups, run.tasks[LONDON].repeats) != (LONDON_WARMUPS, LONDON_REPEATS):
        raise typer.BadParameter(f"{name} has not run {LONDON}^{LONDON_WARMUPS}+{LONDON_REPEATS}")
    dataset, repeats, decode = LONDON, LONDON_REPEATS, london
    if run.ready.specdec:
        ran = {mixture: [task for task in run.responses if task.startswith(f"{mixture}/")] for mixture in MIXTURES}
        whole = [mixture for mixture, tasks in ran.items() if len(tasks) == MIXTURES[mixture]]
        if len(whole) != 1:
            has = ", ".join(f"{len(tasks)}/{MIXTURES[mixture]} of {mixture}" for mixture, tasks in ran.items())
            raise typer.BadParameter(f"{name} has to have run the whole of one mixture, and has {has}")
        (dataset,) = whole
        counts = {run.tasks[task].repeats for task in ran[dataset]}
        if len(counts) != 1:
            raise typer.BadParameter(f"{name} has run the tasks of {dataset} different numbers of times")
        (repeats,) = counts
        decode = summarize(run.responses[task] for task in ran[dataset])
    return Measured(
        speculative=run.ready.specdec,
        decode_dataset=dataset,
        decode_repeats=repeats,
        decode_tps=decode.decode_tps,
        tokens_per_forward_pass=decode.tokens_per_forward_pass,
        forwards_per_second=decode.forwards_per_second,
        prefill_tps=london.prefill_tps,
        time_to_first_token=london.time_to_first_token,
        memory_used=london.memory_used,
    )


def result(run: Run) -> Result:
    return Result(
        device=run.device,
        engine=run.engine,
        model=run.model,
        source=run.config["model"],
        speculator=run.config.get("draft_model"),
        measured=None if run.failures else measured(run),
        failure=run.failure() if run.failures else None,
    )


def main(
    source: Annotated[str, typer.Argument(metavar="FILE|tag:POOL/NAME")],
    target: Annotated[Path, typer.Argument(metavar="FILE")],
) -> None:
    results = [result(run) for run in load(source)]
    target.write_bytes(TypeAdapter(list[Result]).dump_json(results, indent=2))


if __name__ == "__main__":
    typer.run(main)
