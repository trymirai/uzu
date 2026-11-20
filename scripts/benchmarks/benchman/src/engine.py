import asyncio
import json
import os
import signal
from asyncio.subprocess import PIPE, Process
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
from typing import Any

from bench import BenchRequest, BenchResponse, ChatMessage, CooldownRequest, CooldownResponse, ReadyResponse
from benchman.subprocess import relay_stderr, subprocess_output
from pydantic import BaseModel


@asynccontextmanager
async def build_engine(source: Path, engine: str) -> AsyncIterator[Path]:
    with TemporaryDirectory() as temporary:
        root = Path(temporary, "root")
        await subprocess_output("nix", "build", "--out-link", str(root), f"path:{source}#benchmarks.engine-{engine}")
        path = root.resolve()
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield path / "bin" / f"engine-{engine}"
    finally:
        os.close(fd)


@asynccontextmanager
async def run_engine(executable: Path, config: dict[str, Any]) -> AsyncIterator[tuple[Process, ReadyResponse]]:
    with TemporaryDirectory() as temporary:
        model = Path(temporary, "model.json")
        model.write_text(json.dumps(config))
        process = await asyncio.create_subprocess_exec(
            executable, model, stdin=PIPE, stdout=PIPE, stderr=PIPE, start_new_session=True, limit=1 << 30
        )
        stderr = asyncio.create_task(relay_stderr(process.stderr))
        try:
            yield process, await read_line(process, ReadyResponse)
        finally:
            process.stdin.close()
            try:
                with suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=10)
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
                await stderr
    if process.returncode:
        raise CalledProcessError(process.returncode, [executable, model])


async def read_line[R: BaseModel](engine: Process, response: type[R]) -> R:
    line = await engine.stdout.readline()
    if not line.endswith(b"\n"):
        raise RuntimeError(f"Engine exited without a {response.__name__} (exit code {await engine.wait()})")
    return response.model_validate_json(line, strict=True, extra="forbid")


REQUEST_TIMEOUT = timedelta(minutes=30)


async def exchange[R: BaseModel](engine: Process, request: BaseModel, response: type[R]) -> R:
    engine.stdin.write(request.model_dump_json(exclude_none=True).encode() + b"\n")
    await engine.stdin.drain()
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT.total_seconds()):
            return await read_line(engine, response)
    except TimeoutError:
        raise RuntimeError(f"Engine did not answer a {type(request).__name__} in {REQUEST_TIMEOUT}") from None


async def warm_up(engine: Process) -> BenchResponse:
    message = ChatMessage(role="user", content="warmup")
    return await run_task(engine, BenchRequest(messages=[message], max_output_tokens=1, reasoning_effort="default"))


# TODO: design a cooldown that waits only on Macs that throttle. A MacBook Pro slowed down at about 80°C.
async def cool_down(engine: Process) -> CooldownResponse:
    return await exchange(engine, CooldownRequest(threshold=80.0, target=70.0), CooldownResponse)


async def run_task(engine: Process, request: BenchRequest) -> BenchResponse:
    return await exchange(engine, request, BenchResponse)
