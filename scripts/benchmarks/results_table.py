"""The results of saved runs as a Markdown table, or with --text as plain text narrow enough for a chat message.

uv run --project benchman python results_table.py head.json --base base.json
"""

from typing import Annotated

import typer
from benchman.report import METRICS, Metric, Run
from benchman.run import load
from benchman.task import summarize

SHOWN = [metric for metric in METRICS if metric.name in ("Decode", "Accepted", "Forwards", "First token", "Memory")]
# What only speculative decoding has.
SPECULATIVE = ("Accepted", "Forwards")


def results(run: Run, base: Run | None, compared: bool, metrics: list[Metric]) -> list[str]:
    """The run's value of each metric, with its change against the base if compared; or what the run lacks."""
    tasks = run.responses.keys() & base.responses.keys() if base else set() if compared else run.responses.keys()
    if run.failures or not tasks:
        lack = "failed" if run.failures else "no base" if run.responses else "missing"
        return [lack, *[""] * (len(metrics) - 1)]
    cells = []
    for metric in metrics:
        ours = metric.value(summarize(run.responses[task] for task in tasks))
        cell = "" if ours is None else format(ours, metric.format)
        if cell and base:
            theirs = metric.value(summarize(base.responses[task] for task in tasks))
            cell += f" {format(ours / theirs - 1, '+.1%'):>6}"
        cells.append(cell)
    return cells


def main(
    runs: Annotated[str, typer.Argument(metavar="FILE|tag:POOL/NAME")],
    base: Annotated[
        str | None, typer.Option(metavar="FILE|tag:POOL/NAME", help="Show the change against these runs.")
    ] = None,
    text: Annotated[bool, typer.Option("--text", help="Plain text in aligned columns.")] = False,
) -> None:
    measured = load(runs)
    bases = {(run.host, run.engine, run.model): run for run in load(base)} if base else {}
    engines = len({run.engine for run in measured}) > 1
    speculative = any(run.ready is not None and run.ready.specdec for run in measured)
    metrics = [metric for metric in SHOWN if speculative or metric.name not in SPECULATIVE]
    partial = any(run.measured() for run in measured)
    names = ["Host", *(["Engine"] if engines else []), "Model", *(["Tasks"] if partial else [])]
    rows = [
        [
            run.host.rpartition("/")[2],
            *([run.engine] if engines else []),
            run.model.removeprefix("trymirai/"),
            *([run.measured()] if partial else []),
            *results(run, bases.get((run.host, run.engine, run.model)), base is not None, metrics),
        ]
        for run in measured
    ]
    if text:
        units = [*[""] * len(names), *(metric.unit for metric in metrics)]
        lines = [[*names, *(metric.name for metric in metrics)], units, *rows]
        widths = [max(map(len, column)) for column in zip(*lines, strict=True)]
        for cells in lines:
            left = [cell.ljust(width) for cell, width in zip(cells[: len(names)], widths, strict=False)]
            right = [cell.rjust(width) for cell, width in zip(cells[len(names) :], widths[len(names) :], strict=True)]
            typer.echo("  ".join([*left, *right]).rstrip())
    else:
        heads = [*names, *(f"{metric.name}<br>{metric.unit}" for metric in metrics)]
        typer.echo(f"| {' | '.join(heads)} |")
        typer.echo(f"|{':--|' * len(names)}{'--:|' * len(metrics)}")
        for cells in rows:
            typer.echo(f"| {' | '.join(' '.join(cell.split()) for cell in cells)} |")
        for run, cells in zip(measured, rows, strict=True):
            if run.failures:
                typer.echo(f"\n{' '.join(cells[: len(names)])}: {' '.join(run.failure().split())}")


if __name__ == "__main__":
    typer.run(main)
