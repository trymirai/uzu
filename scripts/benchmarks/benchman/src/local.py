from contextlib import AsyncExitStack
from pathlib import Path

from benchman.device import read_device
from benchman.engine import build_engine, cool_down, run_engine, run_task, warm_up
from benchman.report import Run
from benchman.source import unpack
from benchman.subprocess import subprocess_output
from benchman.task import TaskResponse, resolve_tasks


async def run_local(run: Run, root: Path, archive: bytes) -> None:
    """Runs the tasks on this Mac. local-direct builds in the checkout at root, local-nix from its archive."""

    async def archived() -> bytes:
        return archive

    try:
        async with AsyncExitStack() as stack:
            run.device = await read_device()
            run.start(f"build {run.engine} engine")
            if run.host == "local-nix":
                source = await stack.enter_async_context(unpack(run.source, archived))
                executable = await stack.enter_async_context(build_engine(source, run.engine))
            else:
                directory = root / "scripts/benchmarks" / f"engine-{run.engine}"
                await subprocess_output(directory / "build.sh")
                executable = directory / "run.sh"
            run.start(f"load {run.config['model']}")
            process, ready = await stack.enter_async_context(run_engine(executable, run.config))
            tasks = resolve_tasks(run.tasks, run.config, ready)
            run.loaded(ready)
            run.start("warmup")
            await warm_up(process)
            for name, request in tasks.items():
                responses = []
                for repeat in range(request.repeats):
                    run.start("cooldown")
                    await cool_down(process)
                    run.start(request.step(name, repeat))
                    responses.append(await run_task(process, request.bench_request()))
                run.responses[name] = TaskResponse.aggregate(responses)
    except Exception as error:  # noqa: BLE001
        run.state, run.completed, run.message = "Failed", False, str(error)
        run.failures = {name: str(error) for name in run.tasks if name not in run.responses}
    else:
        run.state, run.completed = "Done", True
    run.announce()
