import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Annotated
from uuid import UUID

import typer
from bench import BenchResponse
from benchman.cli import app, console
from benchman.flow import RUN_RESULT_ARTIFACT_TYPE, RunResult, TaskRequest
from prefect.client.orchestration import SyncPrefectClient, get_client
from prefect.client.schemas.filters import (
    ArtifactFilter,
    ArtifactFilterFlowRunId,
    ArtifactFilterType,
    FlowFilter,
    FlowFilterName,
    FlowRunFilter,
    FlowRunFilterId,
    FlowRunFilterName,
    TaskRunFilter,
    TaskRunFilterState,
    TaskRunFilterStateType,
)
from prefect.client.schemas.objects import FlowRun, StateType, TaskRun
from prefect.client.schemas.sorting import ArtifactSort
from prefect.utilities.urls import url_for
from rich import box
from rich.console import Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


class RunView:
    def __init__(self, run: FlowRun) -> None:
        self.run = run
        self.running: list[TaskRun] = []
        self.spinner = Spinner("dots", style="status.spinner")

    def __rich__(self) -> Spinner | Text:
        line = Text(f"{self.run.state_name or 'Submitted'} ")
        line.append(self.run.name, style=f"link {url_for('flow-run', obj_id=self.run.id)}")
        if self.run.state is not None and self.run.state.is_final():
            return line
        if self.running:
            task_run = max(self.running, key=lambda task_run: task_run.start_time)
            elapsed = (datetime.now(UTC) - task_run.start_time).total_seconds()
            line.append(f" · {task_run.name} {elapsed:.1f} s")
        self.spinner.update(text=line)
        return self.spinner


def watch(flow_run_ids: list[UUID]) -> None:
    flow_run_filter = FlowRunFilter(id=FlowRunFilterId(any_=flow_run_ids))
    running = TaskRunFilter(state=TaskRunFilterState(type=TaskRunFilterStateType(any_=[StateType.RUNNING])))
    with get_client(sync_client=True) as client:
        views = {run.id: RunView(run) for run in client.read_flow_runs(flow_run_filter=flow_run_filter)}
        with Live(
            Group(*(views[flow_run_id] for flow_run_id in flow_run_ids)),
            console=console,
            refresh_per_second=20,
            transient=True,
        ):
            while not all(view.run.state is not None and view.run.state.is_final() for view in views.values()):
                time.sleep(1)
                task_runs = client.read_task_runs(flow_run_filter=flow_run_filter, task_run_filter=running)
                for run in client.read_flow_runs(flow_run_filter=flow_run_filter):
                    views[run.id].run = run
                    views[run.id].running = [task_run for task_run in task_runs if task_run.flow_run_id == run.id]


def read_results(client: SyncPrefectClient, flow_run_id: UUID) -> dict[str, list[BenchResponse]]:
    artifact_filter = ArtifactFilter(
        flow_run_id=ArtifactFilterFlowRunId(any_=[flow_run_id]),
        type=ArtifactFilterType(any_=[RUN_RESULT_ARTIFACT_TYPE]),
    )
    artifacts = []
    while page := client.read_artifacts(
        artifact_filter=artifact_filter, sort=ArtifactSort.CREATED_DESC, offset=len(artifacts)
    ):
        artifacts += page
    results: dict[str, list[BenchResponse]] = {}
    for result in sorted(
        (RunResult.model_validate(artifact.data) for artifact in reversed(artifacts)), key=lambda result: result.repeat
    ):
        results.setdefault(result.task, []).append(result.response)
    return results


# The catalog key a flow run was asked for, which the engine's own name for the model doesn't tell.
MODEL_LABEL = "benchman.model"


def accepted_length(responses: list[BenchResponse]) -> str:
    lengths = [
        response.tokens_per_forward_pass for response in responses if response.tokens_per_forward_pass is not None
    ]
    return f"{mean(lengths):.2f}" if lengths else ""


# Name, unit, which of two values is the better one, and the value to show for a flow run's responses.
METRICS: list[tuple[str, str, Callable[..., float], Callable[[list[BenchResponse]], str]]] = [
    ("Decode", "tokens/s", max, lambda responses: f"{mean(response.decode_tps for response in responses):.1f}"),
    ("Accepted", "tokens", max, accepted_length),
    ("Prefill", "tokens/s", max, lambda responses: f"{mean(response.prefill_tps for response in responses):.0f}"),
    ("First token", "s", min, lambda responses: f"{mean(response.time_to_first_token for response in responses):.2f}"),
    (
        "Memory",
        "GiB",
        min,
        lambda responses: f"{max(response.memory_used for response in responses) / 2**30:.2f}",
    ),
]


@dataclass
class Summary:
    run: FlowRun
    status: str
    complete: bool
    metrics: list[str]


def show(flow_run_ids: list[UUID]) -> int:
    """Prints what the flow runs have measured so far and returns the exit code that tells how far they got.

    Runs of one model on one queue share a row, with a line for each engine.
    """
    groups: list[tuple[str, str, dict[str, Summary]]] = []
    with get_client(sync_client=True) as client:
        for flow_run_id in flow_run_ids:
            run = client.read_flow_run(flow_run_id)
            responses = [response for runs in read_results(client, flow_run_id).values() for response in runs]
            planned = sum(TaskRequest.model_validate(task).repeats for task in run.parameters["tasks"].values())
            summary = Summary(
                run=run,
                status=f"{run.state_name} {len(responses)}/{planned}",
                complete=run.state.is_completed() and len(responses) == planned,
                metrics=[measure(responses) if responses else "" for *_, measure in METRICS],
            )
            if MODEL_LABEL not in run.labels:
                raise typer.BadParameter(f"{run.name} has no {MODEL_LABEL} label, so `benchman run` did not submit it")
            key = (run.labels[MODEL_LABEL], f"{run.work_pool_name}/{run.work_queue_name}")
            engine = run.parameters["engine"]
            engines = next((engines for *group, engines in groups if (*group,) == key and engine not in engines), None)
            if engines is None:
                groups.append((*key, engines := {}))
            engines[engine] = summary
    summaries = [summary for *_, engines in groups for summary in engines.values()]
    measured = [index for index in range(len(METRICS)) if any(summary.metrics[index] for summary in summaries)]

    # A table of runs that all measured everything has no status to tell.
    unfinished = not all(summary.complete for summary in summaries)
    table = Table(show_lines=True, box=box.ROUNDED)
    table.add_column("Model", vertical="middle", style="magenta", overflow="fold")
    table.add_column("Queue", vertical="middle", style="magenta", overflow="fold")
    table.add_column("Flow run")
    if unfinished:
        table.add_column("Status", vertical="middle")
    table.add_column("Engine")
    for index in measured:
        table.add_column("\n".join(METRICS[index][:2]), justify="right", style="cyan")
    for model, queue, engines in groups:
        row: list[str | Text] = [model, queue, "\n".join(summary.run.name for summary in engines.values())]
        if unfinished:
            statuses = {summary.status for summary in engines.values()}
            shared = len(statuses) == 1 and all(summary.complete for summary in engines.values())
            row.append(statuses.pop() if shared else "\n".join(summary.status for summary in engines.values()))
        row.append("\n".join(engines))
        for index in measured:
            cells = [summary.metrics[index] for summary in engines.values()]
            values = {float(cell) for cell in cells if cell}
            # The best value of an engine comparison is underlined.
            best = METRICS[index][2](values) if len(values) > 1 else None
            row.append(
                Text("\n").join(Text(cell, style="underline" if cell and float(cell) == best else "") for cell in cells)
            )
        table.add_row(*row)
    console.print(table)

    runs = [summary.run for summary in summaries]
    for run in runs:
        if run.state.is_final() and not run.state.is_completed():
            console.print(f"{run.name}: {run.state.message}", style="red", highlight=False)
    if all(run.state.is_completed() for run in runs):
        return 0
    if all(run.state.is_final() for run in runs):
        return 1
    return os.EX_TEMPFAIL


@app.command("result", hidden=True)
@app.command()
def results(
    flow_runs: Annotated[list[str], typer.Argument(help="Names or IDs of benchmark flow runs.")],
    wait: Annotated[bool, typer.Option("--wait", "-w", help="Wait for the flow runs to finish.")] = False,
) -> None:
    """Show what benchmark flow runs measured.

    Exits with 0 when all of them completed, 1 when one failed, and 75 when some are still to finish.
    """

    flow_run_ids = []
    with get_client(sync_client=True) as client:
        for flow_run in flow_runs:
            try:
                flow_run_ids.append(UUID(flow_run))
            except ValueError:
                named = client.read_flow_runs(
                    flow_filter=FlowFilter(name=FlowFilterName(any_=["benchmark"])),
                    flow_run_filter=FlowRunFilter(name=FlowRunFilterName(any_=[flow_run])),
                )
                if len(named) != 1:
                    raise typer.BadParameter(f"{len(named)} benchmark flow runs are named {flow_run}") from None
                flow_run_ids.append(named[0].id)
    if wait:
        watch(flow_run_ids)
    raise typer.Exit(show(flow_run_ids))
