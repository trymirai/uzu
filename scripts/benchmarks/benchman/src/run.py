import asyncio
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from benchman.cli import app, console
from benchman.datasets import LOADERS, load_dataset
from benchman.engine import engine_config
from benchman.flow import TaskRequest, benchmark
from benchman.results import MODEL_LABEL, show, watch
from benchman.storage import upload_nar
from prefect.client.orchestration import get_client
from prefect.flows import bind_flow_to_infrastructure
from prefect.results import resolve_result_storage
from prefect.states import Cancelling
from prefect.utilities.urls import url_for
from prefect.workers.process import ProcessWorker
from typer.core import TyperCommand


class RepeatableOptionsCommand(TyperCommand):
    """Lets a repeatable option take every value up to the next option: `-m a b` is `-m a -m b`."""

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        repeatable = {flag for parameter in self.params if parameter.multiple for flag in parameter.opts}
        expanded = []
        flag = None
        for arg in args:
            if arg.startswith("-"):
                flag = arg if arg in repeatable else None
                if flag is None:
                    expanded.append(arg)
            elif flag is not None:
                expanded += [flag, arg]
            else:
                expanded.append(arg)
        return super().parse_args(ctx, expanded)


def split(values: list[str]) -> list[str]:
    return [item for value in values for item in value.split(",")]


@app.command(cls=RepeatableOptionsCommand)
def run(
    on: Annotated[
        list[str], typer.Option("--on", "-o", metavar="POOL/QUEUE...", help="Work pools' queues to run on each of.")
    ],
    models: Annotated[
        list[str],
        typer.Option(
            "--model",
            "--models",
            "-m",
            metavar="MODEL...",
            help="Models as keyed in the engines' models.json; trymirai/ is implied.",
        ),
    ],
    engines: Annotated[
        list[str],
        typer.Option("--engine", "--engines", "-e", metavar="ENGINE...", help="Engines to run each model with."),
    ] = ["uzu"],  # noqa: B006
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "--datasets",
            "-d",
            metavar="NAME[/KEY]...",
            help="Datasets, or single prompts of them, to run. Defaults to mirai-mixture-fast for models that decode"
            " speculatively and to oneoff/london-summary for the others.",
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", "-s", help="Sampling seed; the engine seeds on its own without one.")
    ] = None,
    background: Annotated[
        bool,
        typer.Option(
            "--background/--foreground", "-b/-f", help="Exit once the flow runs are submitted, or wait for results."
        ),
    ] = False,
) -> None:
    """Benchmark this checkout, with one flow run per queue, engine and model.

    Options that take several values take them separated by spaces or commas, or given again.
    """

    queues = [queue.partition("/") for queue in split(on)]
    if not all(work_pool and work_queue for work_pool, _, work_queue in queues):
        raise typer.BadParameter("Expected POOL/QUEUE", param_hint="--on")

    git = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True)
    root = Path(git.stdout.strip())
    configs = {}
    for engine in split(engines):
        for model in split(models):
            model = model if "/" in model else f"trymirai/{model}"
            try:
                configs[engine, model] = engine_config(root, engine, model)
            except FileNotFoundError:
                raise typer.BadParameter(f"There is no engine-{engine}", param_hint="--engine") from None
            except KeyError:
                raise typer.BadParameter(f"engine-{engine}/models.json has no {model}", param_hint="--model") from None

    selections = {
        cell: split(datasets or ["mirai-mixture-fast" if config["specdec"] else "oneoff/london-summary"])
        for cell, config in configs.items()
    }
    with console.status("Loading datasets"):
        tasks = {}
        for selection in {selection for chosen in selections.values() for selection in chosen}:
            name, _, prompt = selection.partition("/")
            if name not in LOADERS:
                raise typer.BadParameter(f"No dataset {name}; there are {', '.join(LOADERS)}", param_hint="--dataset")
            requests = asyncio.run(load_dataset(name))
            if prompt:
                if prompt not in requests:
                    raise typer.BadParameter(f"{name} has no prompt {prompt}", param_hint="--dataset")
                requests = {prompt: requests[prompt]}
            tasks[selection] = {
                f"{name}/{key}": TaskRequest(**(request.model_dump() | {"seed": seed}))
                for key, request in requests.items()
            }

    with console.status("Uploading archive"):
        storages = {}
        for work_pool in {work_pool for work_pool, _, _ in queues}:
            with get_client(sync_client=True) as client:
                storage_id = client.read_work_pool(work_pool).storage_configuration.default_result_storage_block_id
            if storage_id is None:
                raise RuntimeError(f"Work pool {work_pool!r} has no default result storage")
            storages[work_pool] = resolve_result_storage(storage_id)
            nar_hash = asyncio.run(upload_nar(root, storages[work_pool]))

    with console.status("Creating flow runs"):
        flow_run_ids = []
        for work_pool, _, work_queue in queues:
            remote = bind_flow_to_infrastructure(
                benchmark.with_options(result_storage=storages[work_pool]),
                work_pool=work_pool,
                work_queue_name=work_queue,
                worker_cls=ProcessWorker,
            )
            for (engine, model), config in configs.items():
                selected = {
                    name: task.model_copy(
                        update={"reasoning_effort": config["reasoning_efforts"][task.reasoning_effort]}
                    )
                    for selection in selections[engine, model]
                    for name, task in tasks[selection].items()
                }
                flow_run_id = remote.submit_to_work_pool(nar_hash, engine, config, selected).flow_run_id
                with get_client(sync_client=True) as client:
                    client.update_flow_run_labels(flow_run_id, {MODEL_LABEL: model})
                flow_run_ids.append(flow_run_id)

    if background:
        with get_client(sync_client=True) as client:
            for flow_run_id in flow_run_ids:
                name = client.read_flow_run(flow_run_id).name
                console.print(name, style=f"link {url_for('flow-run', obj_id=flow_run_id)}", highlight=False)
        return
    try:
        watch(flow_run_ids)
    except KeyboardInterrupt:
        with get_client(sync_client=True) as client:
            for flow_run_id in flow_run_ids:
                client.set_flow_run_state(flow_run_id, Cancelling())
        try:
            watch(flow_run_ids)
        except KeyboardInterrupt:
            pass
        raise typer.Exit(130) from None
    raise typer.Exit(show(flow_run_ids))
