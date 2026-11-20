import asyncio
import io
import logging
import re
from collections.abc import AsyncIterator, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager, closing, redirect_stderr
from datetime import UTC, datetime, timedelta
from functools import cache
from threading import Lock
from typing import Any, TypeVar
from uuid import UUID, uuid4

from benchman.device import read_device
from benchman.engine import build_engine, cool_down, run_engine, run_task, warm_up
from benchman.nix_path_hack import add_nix_to_path
from benchman.pickle_by_value_hack import pickle_benchman_by_value
from benchman.prefect_bundle_quiet_hack import quieten_bundle_dependencies
from benchman.report import Run
from benchman.source import unpack
from benchman.storage import THREADS, Records, Storage, digest
from benchman.task import Failure, Outcome, TaskRecord, TaskResponse, resolve_tasks
from prefect import Task, flow, get_run_logger
from prefect.artifacts import acreate_markdown_artifact
from prefect.bundles import convert_step_to_command, create_bundle_for_flow_run
from prefect.cache_policies import NO_CACHE
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterId,
    TaskRunFilter,
    TaskRunFilterState,
    TaskRunFilterStateType,
)
from prefect.client.schemas.objects import FlowRun, StateType
from prefect.context import FlowRunContext
from prefect.results import resolve_result_storage
from prefect.states import Cancelling, Pending, Scheduled
from prefect.task_engine import AsyncTaskRunEngine
from prefect.utilities.callables import get_call_parameters
from prefect.utilities.processutils import command_to_string
from prefect.utilities.urls import url_for

T = TypeVar("T")


class TaskBlock:
    output: Any = None

    def set_output(self, output: T) -> T:
        self.output = output
        return output


async def block() -> None:
    pass


@asynccontextmanager
async def task(name: str, *upstream: Any) -> AsyncIterator[TaskBlock]:
    engine = AsyncTaskRunEngine(
        task=Task(block, name=name, task_run_name=name, cache_policy=NO_CACHE, persist_result=False),
        wait_for=list(upstream),
    )
    async with engine.start(), engine.run_context(), engine.transaction_context() as transaction:
        body = TaskBlock()
        yield body
        await engine.handle_success(body.output, transaction)
    await engine.result()


OVERWRITTEN = re.compile(r"\r|\x1b\[[0-9;]*K")
ESCAPES = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class LoggedLines(io.TextIOBase):
    """Logs the lines written to it, each as a terminal would have left it."""

    def __init__(self, logger: logging.Logger | logging.LoggerAdapter) -> None:
        self.logger = logger
        self.pending = ""

    def write(self, text: str) -> int:
        *lines, self.pending = (self.pending + text).split("\n")
        for line in lines:
            if shown := ESCAPES.sub("", OVERWRITTEN.split(line)[-1]).strip():
                self.logger.info(shown)
        return len(text)

    def close(self) -> None:
        self.write("\n")


@flow(persist_result=False)
async def benchmark(nar_hash: str, engine: str, config: dict[str, Any], tasks: str, run: str) -> None:
    """Measures the tasks whose records name this flow run, and stores how each ended.

    A failure is stored for the task that was being measured, or for every task if none was yet. A task that
    another flow run takes over meanwhile is left to it.
    """
    add_nix_to_path()
    context = FlowRunContext.get()
    storage = Storage(context.result_store.result_storage)
    requests = await asyncio.to_thread(storage.read_tasks, tasks)
    records = Records(storage, run, requests)

    def own(record: TaskRecord | None) -> bool:
        return record is not None and record.flow_run_id == context.flow_run.id and record.outcome is None

    await records.refresh()
    claimed = {name for name, record in records.stored.items() if own(record)}
    if not claimed:
        return
    device = await read_device()
    ready = None
    measuring = None

    async def conclude(name: str, result: TaskResponse | Failure | None) -> None:
        outcome = Outcome(device=device, ready=ready, result=result)
        # A tag may be added to the record until its outcome is stored.
        while own(record := await asyncio.to_thread(records.read, name)):
            concluded = record.model_copy(update={"outcome": outcome})
            if await asyncio.to_thread(records.write, name, concluded, record):
                for tag in record.tags:
                    await asyncio.to_thread(records.copy, name, concluded, tag)
                break
        claimed.discard(name)

    try:
        async with AsyncExitStack() as stack:
            lines = stack.enter_context(closing(LoggedLines(get_run_logger())))
            stack.enter_context(redirect_stderr(lines))
            async with task("download archive") as block:
                archive = unpack(nar_hash, lambda: asyncio.to_thread(storage.read_archive, nar_hash))
                source = block.set_output(await stack.enter_async_context(archive))
            async with task(f"build {engine} engine", source) as block:
                executable = block.set_output(await stack.enter_async_context(build_engine(source, engine)))
            async with task(f"load {config['model']}", executable) as block:
                process, ready = await stack.enter_async_context(run_engine(executable, config))
                block.set_output(process)
            requests = resolve_tasks(
                {name: request for name, request in requests.items() if name in claimed}, config, ready
            )
            for name in claimed - requests.keys():
                await conclude(name, None)
            async with task("warmup", process) as block:
                previous = block.set_output(await warm_up(process))
            for name, request in requests.items():
                if not own(await asyncio.to_thread(records.read, name)):
                    claimed.discard(name)
                    continue
                measuring = name
                for warmup in range(request.warmups):
                    async with task(request.warmup_step(name, warmup), previous) as block:
                        previous = block.set_output(await run_task(process, request.bench_request()))
                responses = []
                for repeat in range(request.repeats):
                    async with task("cooldown", previous) as block:
                        cooled = block.set_output(await cool_down(process))
                    async with task(request.step(name, repeat), cooled) as block:
                        response = await run_task(process, request.bench_request())
                        summary = f"- Decode: {response.decode_tps:.1f} tokens/s\n"
                        if response.tokens_per_forward_pass is not None:
                            summary += f"- Mean accepted length: {response.tokens_per_forward_pass:.2f}\n"
                            summary += f"- Forward passes: {response.forwards_per_second:.2f} per second\n"
                        summary += (
                            f"- Prefill: {response.prefill_tps:.0f} tokens/s\n"
                            f"- Time to first token: {response.time_to_first_token:.2f} s\n"
                            f"- Memory: {response.memory_used / 2**30:.2f} GiB\n"
                        )
                        await acreate_markdown_artifact(markdown=summary, description=name)
                        previous = block.set_output(response)
                        responses.append(response)
                await conclude(name, TaskResponse.aggregate(responses))
    except Exception as error:
        for name in [measuring] if measuring else list(claimed):
            await conclude(name, Failure(message=str(error)))
        raise


@cache
def pool_storage(work_pool: str) -> Storage:
    with get_client(sync_client=True) as client:
        storage_id = client.read_work_pool(work_pool).storage_configuration.default_result_storage_block_id
    if storage_id is None:
        raise RuntimeError(f"Work pool {work_pool!r} has no default result storage")
    return Storage(resolve_result_storage(storage_id))


def records(run: Run) -> Records:
    key = digest({"source": run.source, "host": run.host, "engine": run.engine, "config": run.config})
    return Records(pool_storage(run.work_pool), key, run.tasks)


def read_tagged(name: str, runs: list[Run]) -> None:
    """Fills the tag's runs with its copies of their task records."""

    async def read() -> list[dict[str, TaskRecord]]:
        return await asyncio.gather(*(records(run).read_all(name, {}) for run in runs))

    for run, copies in zip(runs, asyncio.run(read()), strict=True):
        run.stored(copies)


ALIVE = {StateType.SCHEDULED, StateType.PENDING, StateType.RUNNING}
# How long a flow run may stay pending while its tasks are claimed and its bundle is uploaded.
RESERVED = timedelta(minutes=30)
# Seconds between looks at the records while waiting.
POLL = 5
# A run's tasks to claim, with the records to claim them over, None for none.
Claims = dict[str, TaskRecord | None]


def foremost(flow_runs: Iterable[FlowRun]) -> FlowRun:
    """Of the flow runs that measure a run's tasks, the one to show: running, else not ended yet."""
    return max(flow_runs, key=lambda flow_run: (not flow_run.state.is_final(), flow_run.state.is_running()))


class Dispatch:
    """The measuring of runs on the fleet: what is stored of them, and the flow runs that measure the rest."""

    def __init__(self, runs: list[Run], tags: Sequence[str]) -> None:
        self.records = {run: records(run) for run in runs}
        self.tags = tags
        self.measuring: dict[UUID, Run] = {}
        # The flow runs that were created here: those that no worker can take yet, and the others.
        self.reserved: set[UUID] = set()
        self.started: set[UUID] = set()

    def inspect(self, runs: Iterable[Run] | None = None, rerun: bool = False) -> dict[Run, Claims]:
        """Fills the runs, by default all, with what is stored, and finds the flow runs that measure the rest.

        Returns each run's tasks to claim: those without a record or with one of a flow run that has ended, and
        with `rerun` all of them.
        """
        runs = list(self.records if runs is None else runs)

        async def refresh() -> None:
            await asyncio.gather(*(self.records[run].refresh() for run in runs))

        asyncio.run(refresh())
        named = {
            record.flow_run_id for run in runs for record in self.records[run].stored.values() if record.outcome is None
        }
        flow_runs: dict[UUID, FlowRun] = {}
        with get_client(sync_client=True) as client:
            known = FlowRunFilter(id=FlowRunFilterId(any_=list(named)))
            while named and (page := client.read_flow_runs(flow_run_filter=known, offset=len(flow_runs))):
                flow_runs |= {flow_run.id: flow_run for flow_run in page}
        stale = datetime.now(UTC) - RESERVED
        alive = {
            flow_run.id
            for flow_run in flow_runs.values()
            if flow_run.state_type in ALIVE and not (flow_run.state.is_pending() and flow_run.state.timestamp < stale)
        }

        def changed(run: Run) -> bool:
            """Whether a flow run that has ended stored an outcome after its record was read."""
            return any(
                self.records[run].read(name) != record
                for name, record in self.records[run].stored.items()
                if record.outcome is None and record.flow_run_id not in alive
            )

        with ThreadPoolExecutor(THREADS) as pool:
            again = [run for run, different in zip(runs, pool.map(changed, runs), strict=True) if different]

        watched = {flow_run_id: run for flow_run_id, run in self.measuring.items() if run not in runs}
        unclaimed = {}
        for run in runs:
            if run in again:
                continue
            stored = self.records[run].stored
            run.stored(stored)
            waiting = []
            ended = []
            claimable = {}
            for name in run.tasks:
                record = stored.get(name)
                awaited = record is not None and record.outcome is None
                flow_run = flow_runs.get(record.flow_run_id) if awaited else None
                measuring = flow_run is not None and flow_run.id in alive
                if rerun or record is None or (awaited and not measuring):
                    claimable[name] = record
                elif measuring:
                    waiting.append(flow_run)
                if flow_run is not None and not measuring:
                    ended.append(flow_run)
            watched |= {flow_run.id: run for flow_run in waiting}
            if claimable:
                unclaimed[run] = claimable
            shown = foremost(waiting).id if waiting else next((record.flow_run_id for record in stored.values()), None)
            run.url = url_for("flow-run", obj_id=shown) if shown else ""
            run.missing = sum(name not in stored for name in run.tasks)
            run.state, run.completed, run.message = "Missing", None, ""
            if waiting:
                run.state = foremost(waiting).state_name
            elif ended and not ended[0].state.is_completed():
                run.state, run.completed, run.message = ended[0].state_name, False, ended[0].state.message or ""
            elif run.failures and not claimable:
                run.state, run.completed, run.message = "Failed", False, run.failure()
            elif not claimable:
                run.state, run.completed = "Done", True
        # At once: the progress display reads it meanwhile.
        self.measuring = watched
        return unclaimed | (self.inspect(again, rerun) if again else {})

    def submit(self, archive: bytes | None, unclaimed: dict[Run, Claims]) -> list[Run]:
        """Creates a flow run per run, for the tasks that it claims over their records.

        Returns the runs with tasks that others claimed first. The archive is of the runs' source, None if that
        is stored already.
        """
        pickle_benchman_by_value()
        quieten_bundle_dependencies()
        bundle_keys = {run: str(uuid4()) for run in unclaimed}
        tasks = {
            run: pool_storage(run.work_pool).write_tasks({name: run.tasks[name] for name in claimable})
            for run, claimable in unclaimed.items()
        }

        bundling = Lock()
        with get_client(sync_client=True) as client, ThreadPoolExecutor(THREADS) as pool:
            steps = {}
            for work_pool, source in {(run.work_pool, run.source) for run in unclaimed}:
                if archive is not None:
                    pool_storage(work_pool).write_archive(source, archive)
                steps[work_pool] = client.read_work_pool(work_pool).storage_configuration

            def reserve(run: Run) -> FlowRun:
                work_pool, _, work_queue = run.host.partition("/")
                execute = convert_step_to_command(steps[work_pool].bundle_execution_step, bundle_keys[run])
                arguments = (run.source, run.engine, run.config, tasks[run], self.records[run].run)
                flow_run = client.create_flow_run(
                    benchmark,
                    parameters=benchmark.serialize_parameters(get_call_parameters(benchmark, arguments, {})),
                    # So that no worker takes it before its tasks are claimed and its bundle is stored.
                    state=Pending(),
                    job_variables={"command": command_to_string(execute)},
                    work_pool_name=work_pool,
                    work_queue_name=work_queue,
                )
                self.reserved.add(flow_run.id)
                return flow_run

            def claim(task: tuple[Run, str]) -> bool:
                run, name = task
                over = unclaimed[run][name]
                # A record that awaits its outcome keeps its tags; a stored one's have their copy.
                kept = over.tags if over and over.outcome is None else []
                record = TaskRecord(flow_run_id=flow_runs[run].id, tags=list(dict.fromkeys([*kept, *self.tags])))
                return self.records[run].write(name, record, over)

            def schedule(run: Run) -> None:
                storage = pool_storage(run.work_pool)
                remote = benchmark.with_options(result_storage=storage.block)
                # One at a time: Prefect has the flow's modules pickled by value only while it makes a bundle,
                # so the end of one would leave another with modules that the workers don't have.
                with bundling:
                    bundle = create_bundle_for_flow_run(flow=remote, flow_run=flow_runs[run])["bundle"]
                # Where the work pool's upload step would store it, without starting that step's process.
                (step,) = steps[run.work_pool].bundle_upload_step.values()
                storage.write_bundle(step["bucket"], bundle_keys[run], bundle)
                client.set_flow_run_state(flow_runs[run].id, Scheduled())
                self.reserved.remove(flow_runs[run].id)
                self.started.add(flow_runs[run].id)

            try:
                flow_runs = dict(zip(unclaimed, pool.map(reserve, unclaimed), strict=True))
                claims = [(run, name) for run, claimable in unclaimed.items() for name in claimable]
                won = {run: [] for run in unclaimed}
                for (run, _), claimed in zip(claims, pool.map(claim, claims), strict=True):
                    won[run].append(claimed)
                for run in unclaimed:
                    if not any(won[run]):
                        client.delete_flow_run(flow_runs[run].id)
                        self.reserved.remove(flow_runs[run].id)
                started = [run for run in unclaimed if any(won[run])]
                list(pool.map(schedule, started))
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise

        for run in started:
            self.measuring[flow_runs[run].id] = run
            run.state, run.completed, run.message, run.missing = "Pending", None, "", 0
            run.url = url_for("flow-run", obj_id=flow_runs[run].id)
        return [run for run in unclaimed if not all(won[run])]

    def tag(self) -> None:
        """Gives the tags a copy of each stored outcome, and has the flow runs that measure the rest give theirs."""

        def give(task: tuple[Records, str]) -> None:
            records, name = task
            record = records.stored[name]
            while record is not None:
                if record.outcome is not None:
                    for tag in self.tags:
                        records.copy(name, record, tag)
                    return
                tags = list(dict.fromkeys([*record.tags, *self.tags]))
                if tags == record.tags or records.write(name, record.model_copy(update={"tags": tags}), record):
                    return
                # Its outcome may have been stored meanwhile.
                record = records.read(name)

        with ThreadPoolExecutor(THREADS) as pool:
            list(pool.map(give, [(records, name) for records in self.records.values() for name in records.stored]))

    async def watch(self) -> None:
        """Updates the runs' progress until no flow run is measuring their tasks."""
        running = TaskRunFilter(state=TaskRunFilterState(type=TaskRunFilterStateType(any_=[StateType.RUNNING])))
        async with get_client() as client:
            while self.measuring:
                watched = FlowRunFilter(id=FlowRunFilterId(any_=list(self.measuring)))
                for _ in range(POLL):
                    await asyncio.sleep(1)
                    task_runs = await client.read_task_runs(flow_run_filter=watched, task_run_filter=running)
                    states: dict[Run, list[FlowRun]] = {}
                    read = 0
                    while page := await client.read_flow_runs(flow_run_filter=watched, offset=read):
                        read += len(page)
                        for flow_run in page:
                            states.setdefault(self.measuring[flow_run.id], []).append(flow_run)
                    for run, own in states.items():
                        shown = foremost(own)
                        run.state, run.url = shown.state_name, url_for("flow-run", obj_id=shown.id)
                        steps = [task_run for task_run in task_runs if self.measuring[task_run.flow_run_id] is run]
                        if steps:
                            step = max(steps, key=lambda task_run: task_run.start_time)
                            run.step, run.since = step.name, step.start_time
                        else:
                            run.since = None
                    if all(flow_run.state.is_final() for own in states.values() for flow_run in own):
                        break
                measured = set(self.measuring.values())
                await asyncio.to_thread(self.inspect, measured)
                for run in measured - set(self.measuring.values()):
                    run.announce()

    def cancel(self, started: bool = False) -> None:
        """Gives up the tasks that flow runs have to measure: of any flow run, or of those started here.

        A flow run goes on with its other tasks, and is cancelled if it has none.
        """

        created = self.reserved | self.started

        def give_up(records: Records) -> set[UUID]:
            """The flow runs that are left without a task."""
            released = {
                record.flow_run_id
                for name, record in list(records.stored.items())
                if record.outcome is None
                and (not started or record.flow_run_id in created)
                and records.delete(name, record)
            }
            # Its other tasks, and those whose records were changed meanwhile.
            kept = [*records.others(), *records.stored.values()] if released else []
            return released - {record.flow_run_id for record in kept if record.outcome is None}

        with ThreadPoolExecutor(THREADS) as pool, get_client(sync_client=True) as client:
            for flow_run_id in set().union(*pool.map(give_up, self.records.values())) - self.reserved:
                client.set_flow_run_state(flow_run_id, Cancelling())
            if started:
                # No worker could take these yet.
                list(pool.map(client.delete_flow_run, self.reserved))
                self.reserved.clear()
