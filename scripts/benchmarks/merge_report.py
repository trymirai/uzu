"""The merge benchmark's report: saved runs against those of the base, as a Markdown table or an image.

uv run --project benchman python merge_report.py head.json --base base.json
uv run --project benchman --group merge-report python merge_report.py head.json --base base.json --image table.png
"""

from collections.abc import Collection
from pathlib import Path
from typing import Annotated, NamedTuple

import typer
from benchman.report import METRICS, Metric, Run
from benchman.run import load
from benchman.task import summarize

# Decode speed is shown as its two factors: the first is steady between runs, the second is not.
FORWARDS, ACCEPTED, *OTHERS = (
    next(metric for metric in METRICS if metric.name == name)
    for name in ("Forwards", "Accepted", "Prefill", "First token", "Memory")
)


class Cell(NamedTuple):
    value: str = ""
    change: str = ""


def amount(metric: Metric, run: Run, tasks: Collection[str]) -> float | None:
    """The run's value of the metric: of what only speculative decoding has, without it what that amounts to."""
    summary = summarize(run.responses[task] for task in tasks)
    measured = metric.value(summary)
    if measured is None and metric.nonspec_placeholder and not run.ready.specdec:
        return metric.nonspec_placeholder(summary)
    return measured


def results(run: Run, base: Run | None) -> list[Cell]:
    """The run's value of each metric with its change against the base, or what the run lacks."""
    tasks = run.responses.keys() & base.responses.keys() if base else set()
    if run.failures or not tasks:
        lack = "failed" if run.failures else "no base" if run.responses else "missing"
        return [Cell(lack), *[Cell()] * (1 + len(OTHERS))]
    cells = []
    for metric in (FORWARDS, ACCEPTED, *OTHERS):
        ours, theirs = amount(metric, run, tasks), amount(metric, base, tasks)
        if ours is None:
            cells.append(Cell())
        else:
            change = "" if theirs is None else format(ours / theirs - 1, "+.1%")
            cells.append(Cell(format(ours, metric.format), change))
    return cells


def markdown(names: list[str], rows: list[list[str]], cells: list[list[Cell]]) -> None:
    heads = [*names, "Decode", *(f"{metric.name}<br>{metric.unit}" for metric in OTHERS)]
    typer.echo(f"| {' | '.join(heads)} |")
    typer.echo(f"|{':--|' * len(names)}{'--:|' * (1 + len(OTHERS))}")
    for row, (forwards, accepted, *others) in zip(rows, cells, strict=True):
        decode = forwards.value
        if accepted.value:
            decode = (
                f"**{forwards.value}&nbsp;fwd/s&nbsp;{forwards.change}**"
                f" ×&nbsp;{accepted.value}&nbsp;tok/fwd&nbsp;{accepted.change}"
            )
        changes = ["&nbsp;".join(cell).removesuffix("&nbsp;") for cell in others]
        typer.echo(f"| {' | '.join([*row, decode, *changes])} |")


BACKGROUND, STRIPE = "#1e1f22", "#26282c"
BRIGHT, TEXT, DIM = "#ffffff", "#c7cad1", "#80848e"
BETTER, WORSE, LACKING = "#57d38c", "#f47b7b", "#e5c453"
# Pixels per point, for sharp text on dense screens.
SCALE = 2
# Percent: a change is grey up to the first, in full colour from the second, and between them the more coloured
# the larger it is.
NOISE, CLEAR = 0.2, 0.6


class Span(NamedTuple):
    text: str = ""
    colour: str = TEXT


class Column(NamedTuple):
    """A column of the image: its parts are aligned each on its own, the first to the left if the column is."""

    name: str
    unit: str
    left: bool
    parts: list[list[Span]]


def change(cell: Cell, metric: Metric) -> Span:
    if not cell.change:
        return Span()
    percent = float(cell.change.rstrip("%"))
    share = min(max((abs(percent) - NOISE) / (CLEAR - NOISE), 0), 1)
    colour = BETTER if (percent > 0) == (metric.best is max) else WORSE
    mixed = (
        round(dim + (full - dim) * share) for dim, full in zip(*map(bytes.fromhex, (DIM[1:], colour[1:])), strict=True)
    )
    return Span(cell.change, f"#{bytes(mixed).hex()}")


def image(path: Path, names: list[str], rows: list[list[str]], cells: list[list[Cell]]) -> None:
    # Of the dependency group merge-report: benchman has no use for it.
    from PIL import Image, ImageDraw, ImageFont

    decode = Column(
        "Decode",
        "",
        False,
        [
            [
                Span(f"{forwards.value} fwd/s", BRIGHT)
                if accepted.value
                else Span(forwards.value, WORSE if forwards.value == "failed" else LACKING)
                for forwards, accepted, *_ in cells
            ],
            [change(forwards, FORWARDS) for forwards, *_ in cells],
            # Pillow's own font has no multiplication sign.
            [Span("x" if accepted.value else "", DIM) for _, accepted, *_ in cells],
            [Span(f"{accepted.value} tok/fwd" if accepted.value else "") for _, accepted, *_ in cells],
            [change(accepted, ACCEPTED) for _, accepted, *_ in cells],
        ],
    )
    columns = [
        *(Column(name, "", True, [[Span(row[index]) for row in rows]]) for index, name in enumerate(names)),
        decode,
        *(
            Column(
                metric.name,
                metric.unit,
                False,
                [[Span(line[index].value, BRIGHT) for line in cells], [change(line[index], metric) for line in cells]],
            )
            for index, metric in enumerate(OTHERS, start=2)
        ),
    ]
    font = ImageFont.load_default(size=15 * SCALE)
    em = font.size
    line, margin, apart, within = 2 * em, em, int(1.6 * em), int(0.4 * em)

    def width(text: str) -> int:
        return round(font.getlength(text))

    parts = [[max(width(span.text) for span in part) for part in column.parts] for column in columns]
    widths = [
        max(sum(part) + within * (len(part) - 1), width(column.name), width(column.unit))
        for column, part in zip(columns, parts, strict=True)
    ]
    size = (2 * margin + sum(widths) + apart * (len(columns) - 1), 2 * margin + line * (2 + len(rows)) - em // 2)
    picture = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(picture)

    def write(span: Span, edge: int, row: float, left: bool) -> None:
        draw.text((edge, margin + int(line * (row + 0.5))), span.text, span.colour, font, "lm" if left else "rm")

    for row in range(1, len(rows), 2):
        top = margin + int(line * (row + 1.75))
        draw.rounded_rectangle((margin // 2, top, size[0] - margin // 2, top + line), em // 3, STRIPE)
    start = margin
    for column, part, wide in zip(columns, parts, widths, strict=True):
        write(Span(column.name, DIM), start, 0, True)
        write(Span(column.unit, DIM), start, 0.75, True)
        # The parts fill the column from its right edge, or the only part of a left one from its left.
        right = start + wide
        for spans, narrow in reversed(list(zip(column.parts, part, strict=True))):
            for row, span in enumerate(spans):
                write(span, start if column.left else right, row + 1.75, column.left)
            right -= narrow + within
        start += wide + apart
    picture.save(path)


def main(
    runs: Annotated[str, typer.Argument(metavar="FILE")],
    base: Annotated[str, typer.Option(metavar="FILE", help="The runs to show the change against.")],
    picture: Annotated[
        Path | None, typer.Option("--image", metavar="FILE", help="Write the table as a PNG instead.")
    ] = None,
) -> None:
    measured = load(runs)
    bases = {(run.host, run.engine, run.model): run for run in load(base)}
    engines = len({run.engine for run in measured}) > 1
    partial = any(run.measured() for run in measured)
    names = ["Host", *(["Engine"] if engines else []), "Model", *(["Tasks"] if partial else [])]
    rows = [
        [
            run.host.rpartition("/")[2].removeprefix("bench-"),
            *([run.engine] if engines else []),
            run.model.removeprefix("trymirai/"),
            *([run.measured()] if partial else []),
        ]
        for run in measured
    ]
    cells = [results(run, bases.get((run.host, run.engine, run.model))) for run in measured]
    if picture:
        image(picture, names, rows, cells)
    else:
        markdown(names, rows, cells)
        for run, row in zip(measured, rows, strict=True):
            if run.failures:
                typer.echo(f"\n{' '.join(row)}: {' '.join(run.failure().split())}")


if __name__ == "__main__":
    typer.run(main)
