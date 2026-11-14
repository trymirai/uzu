import asyncio
import json
import os
import signal
from asyncio.subprocess import PIPE, Process
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
from typing import Any

from bench import BenchRequest, BenchResponse, CooldownRequest, CooldownResponse, ReadyResponse
from benchman.subprocess import subprocess_output
from pydantic import BaseModel


@asynccontextmanager
async def build_engine(source: Path, engine: str) -> AsyncIterator[Path]:
    with TemporaryDirectory() as temporary:
        root = Path(temporary, "root")
        await subprocess_output("nix", "build", "--out-link", str(root), f"path:{source}#bench-{engine}")
        path = root.resolve()
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield path / "bin" / f"bench-{engine}"
    finally:
        os.close(fd)


def engine_config(source: Path, engine: str, model: str) -> dict[str, Any]:
    return json.loads((source / "scripts/benchmarks" / f"engine-{engine}" / "models.json").read_text())[model]


@asynccontextmanager
async def run_engine(executable: Path, config: dict[str, Any]) -> AsyncIterator[Process]:
    with TemporaryDirectory() as temporary:
        model = Path(temporary, "model.json")
        model.write_text(json.dumps(config))
        # A response line carries the whole generated text, beyond StreamReader's default 64 KiB line limit.
        process = await asyncio.create_subprocess_exec(
            executable, model, stdin=PIPE, stdout=PIPE, start_new_session=True, limit=1 << 30
        )
        try:
            await read_line(process, ReadyResponse)
            yield process
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
    if process.returncode:
        raise CalledProcessError(process.returncode, [executable, model])


async def read_line[R: BaseModel](engine: Process, response: type[R]) -> R:
    line = await engine.stdout.readline()
    if not line.endswith(b"\n"):
        raise RuntimeError(f"Engine exited without a {response.__name__} (exit code {await engine.wait()})")
    return response.model_validate_json(line, strict=True, extra="forbid")


async def send(engine: Process, request: BaseModel) -> None:
    engine.stdin.write(request.model_dump_json(exclude_none=True).encode() + b"\n")
    await engine.stdin.drain()


async def cool_down(engine: Process, request: CooldownRequest) -> CooldownResponse:
    await send(engine, request)
    return await read_line(engine, CooldownResponse)


async def run_task(engine: Process, request: BenchRequest) -> BenchResponse:
    await send(engine, request)
    return await read_line(engine, BenchResponse)
