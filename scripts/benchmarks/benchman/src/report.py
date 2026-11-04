import os
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any, NamedTuple

from bench import ReadyResponse
from benchman.device import Device
from benchman.rich_live_resize_hack import count_rows_after_resize
from benchman.task import Failure, Summary, TaskRecord, TaskRequest, TaskResponse, summarize
from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass
from rich import box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass(eq=False, config=ConfigDict(arbitrary_types_allowed=True))
class Run:
    """One engine with one model on one host. Fields that are not excluded are saved."""

    source: str
    host: str
    engine: str
    model: str
    config: dict[str, Any]
    tasks: dict[str, TaskRequest]
    device: Device | None = None
    ready: ReadyResponse | None = None
    responses: dict[str, TaskResponse] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)
    label: str = Field(default="", exclude=True)
    url: str = Field(default="", exclude=True)
    state: str = Field(default="Pending", exclude=True)
    # None until the run has ended.
    completed: bool | None = Field(default=None, exclude=True)
    message: str = Field(default="", exclude=True)
    step: str = Field(default="", exclude=True)
    since: datetime | None = Field(default=None, exclude=True)
    spinner: Spinner = Field(default_factory=lambda: Spinner("dots", style="status.spinner"), exclude=True)

    @property
    def work_pool(self) -> str:
        return self.host.partition("/")[0]

    def start(self, step: str) -> None:
        self.state, self.step, self.since = "Running", step, datetime.now(UTC)

    def loaded(self, ready: ReadyResponse) -> None:
        self.ready = ready
        self.tasks = {name: task for name, task in self.tasks.items() if task.specdec in (None, ready.specdec)}

    def stored(self, records: dict[str, TaskRecord]) -> None:
        """Takes what the records of its tasks hold."""
        outcomes = {name: record.outcome for name, record in records.items() if record.outcome is not None}
        results = {name: outcome.result for name, outcome in outcomes.items()}
        self.responses = {name: result for name, result in results.items() if isinstance(result, TaskResponse)}
        self.failures = {name: result.message for name, result in results.items() if isinstance(result, Failure)}
        if outcomes:
            self.device = next(iter(outcomes.values())).device
        if ready := next((outcome.ready for outcome in outcomes.values() if outcome.ready is not None), None):
            self.loaded(ready)

    def failure(self) -> str:
        """The stored failure to show: of the first task, with its name unless it is every task's."""
        name, message = next(iter(self.failures.items()))
        shared = len(self.failures) == len(self.tasks) and set(self.failures.values()) == {message}
        return message if shared else f"{name}: {message}"

    def name(self) -> Text:
        return Text(f"{self.host} {self.engine} {self.model}", style=f"link {self.url}" if self.url else "")

    def measured(self) -> str:
        """How many of the tasks have results, once any has."""
        return f"{len(self.responses)}/{len(self.tasks)}" if self.responses else ""

    def progress(self) -> Spinner:
        step = f"{self.step} {(datetime.now(UTC) - self.since).total_seconds():.1f} s" if self.since else ""
        parts = (f" · {part}" for part in (self.measured(), step) if part)
        self.spinner.update(text=Text.assemble(self.name(), *parts))
        return self.spinner

    def announce(self) -> None:
        """Prints how the run ended; above the progress, if that is shown."""
        style = {True: "bold green", False: "bold red", None: "bold yellow"}[self.completed]
        measured = f" {self.measured()}" if self.responses and not self.completed else ""
        console.print(Text.assemble((f"{self.state:>9} ", style), self.name(), measured), highlight=False)


RUNS = TypeAdapter(list[Run])


def balanced(items: list[str], width: int) -> list[str]:
    """The items on as few lines as fit the width, each line with about as many of them."""
    for count in range(1, len(items) + 1):
        ends = [len(items) * line // count for line in range(count + 1)]
        lines = [" · ".join(items[start:end]) for start, end in pairwise(ends)]
        if all(cell_len(line) <= width for line in lines):
            break
    return lines


# States of a run that has not ended, in the order of the counts.
NEAREST = ("Running", "Submitting", "Late", "Scheduled", "Pending")


class Progress:
    """The runs that are running, and below them what else is left."""

    def __init__(self, runs: list[Run], hint: Callable[[], str] | None) -> None:
        self.runs = runs
        self.hint = hint

    def __rich__(self) -> Group:
        unfinished = [run for run in self.runs if run.completed is None and run.state != "Missing"]
        waiting = [run for run in unfinished if run.state != "Running"]
        lines: list[Spinner | Text] = [run.progress() for run in unfinished if run.state == "Running"]
        if waiting:

            def stage(run: Run) -> tuple[bool, bool, int, str]:
                """The nearest to a result first, then what ended without one, then what is missing."""
                nearest = NEAREST.index(run.state) if run.state in NEAREST else len(NEAREST)
                return run.state == "Missing", run.completed is not None, nearest, run.state

            done = [run for run in self.runs if run.completed]
            states = Counter(run.state for run in sorted(self.runs, key=stage) if not run.completed)
            hosts = Counter(run.host for run in waiting)
            for counted in (states.items(), hosts.most_common()):
                counts = [f"{name} {count}" for name, count in counted]
                lines += [Text(line, style="dim") for line in balanced(counts, console.width)]
            tasks = sum(len(run.tasks) for run in self.runs)
            measured = sum(len(run.responses) for run in self.runs)
            lines.append(Text(f"Done {len(done)}/{len(self.runs)} runs, {measured}/{tasks} tasks", style="dim"))
        if self.hint:
            lines.append(Text(self.hint(), style="dim"))
        return Group(*lines)


def progress(runs: list[Run], hint: Callable[[], str] | None = None) -> Live:
    return count_rows_after_resize(Live(Progress(runs, hint), console=console, refresh_per_second=60, transient=True))


class Metric(NamedTuple):
    name: str
    unit: str
    best: Callable[..., float]
    format: str
    value: Callable[[Summary], float | None]


METRICS = [
    Metric("Decode", "tokens/s", max, ".1f", lambda summary: summary.decode_tps),
    Metric("Accepted", "tokens", max, ".2f", lambda summary: summary.tokens_per_forward_pass),
    Metric("Forwards", "passes/s", max, ".2f", lambda summary: summary.forwards_per_second),
    Metric("Prefill", "tokens/s", max, ".0f", lambda summary: summary.prefill_tps),
    Metric("First token", "s", min, ".2f", lambda summary: summary.time_to_first_token),
    Metric("Memory", "GiB", min, ".2f", lambda summary: summary.memory_used / 2**30),
]


def report(runs: list[Run], baseline: str | None) -> int:
    """Prints the results table and returns the exit code: 0 all measured, 1 a run failed, 75 results missing.

    A row is a model on a host, with a line per engine and label. Values show their change against the baseline
    label over the tasks both have.
    """
    rows: list[tuple[str, str, dict[tuple[str, str], Run]]] = []
    for run in runs:
        line = (run.engine, run.label)
        fitting = (lines for model, host, lines in rows if (model, host) == (run.model, run.host) and line not in lines)
        lines = next(fitting, None)
        if lines is None:
            rows.append((run.model, run.host, lines := {}))
        lines[line] = run

    def value(run: Run, metric: Metric, tasks: Iterable[str] | None = None) -> float | None:
        responses = [run.responses[name] for name in (run.responses if tasks is None else tasks)]
        return metric.value(summarize(responses)) if responses else None

    metrics = [metric for metric in METRICS if any(value(run, metric) is not None for run in runs)]
    status = {run: f"{run.state} {run.measured()}".strip() for run in runs}
    complete = {run: bool(run.completed) and len(run.responses) == len(run.tasks) for run in runs}
    unfinished = not all(complete.values())
    labelled = len({run.label for run in runs}) > 1

    table = Table(show_lines=True, box=box.ROUNDED)
    table.add_column("Model", vertical="middle", style="magenta", overflow="fold", min_width=10)
    table.add_column("Host", vertical="middle", style="magenta", overflow="fold", min_width=10)
    if unfinished:
        table.add_column("Status", vertical="middle", no_wrap=True)
    # Cells hold a line per run, which wrapping would misalign.
    table.add_column("Engine", no_wrap=True)
    if labelled:
        table.add_column("Results", no_wrap=True)
    for metric in metrics:
        table.add_column(f"{metric.name}\n{metric.unit}", justify="right", style="cyan", no_wrap=True)
    for model, host, lines in rows:
        engines = list(dict.fromkeys(engine for engine, _ in lines))
        lines = dict(sorted(lines.items(), key=lambda line: engines.index(line[0][0])))
        row: list[str | Text] = [model, host]
        if unfinished:
            statuses = {status[run] for run in lines.values()}
            shared = len(statuses) == 1 and all(complete[run] for run in lines.values())
            row.append(statuses.pop() if shared else "\n".join(status[run] for run in lines.values()))
        row.append("\n".join(engine for engine, _ in lines))
        if labelled:
            row.append("\n".join(label for _, label in lines))
        for metric in metrics:
            shown = {
                run: format(measured, metric.format)
                for run in lines.values()
                if (measured := value(run, metric)) is not None
            }
            best = metric.best(map(float, shown.values())) if len(set(shown.values())) > 1 else None
            cells = []
            for (engine, _), run in lines.items():
                cell = Text(shown.get(run, ""), style="underline" if run in shown and float(shown[run]) == best else "")
                base = lines.get((engine, baseline))
                if base is not None and base is not run:
                    shared = run.responses.keys() & base.responses.keys()
                    ours, theirs = value(run, metric, shared), value(base, metric, shared)
                    if ours is not None and theirs:
                        cell.append(f" {ours / theirs - 1:+.1%}", style="dim")
                cells.append(cell)
            row.append(Text("\n").join(cells))
        table.add_row(*row)
    # Never narrower than the minimum, or the lines of a row misalign.
    console.print(table, width=max(console.width, console.measure(table).minimum), crop=False)

    for run in runs:
        if run.completed is False and run.message:
            console.print(f"{run.host} {run.engine} {run.model}: {run.message}", style="red", highlight=False)
    if all(complete.values()):
        return 0
    if all(run.completed is not None for run in runs) and not all(run.completed for run in runs):
        return 1
    return os.EX_TEMPFAIL
