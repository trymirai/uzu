import asyncio
import sys
from asyncio import StreamReader
from asyncio.subprocess import DEVNULL, PIPE
from pathlib import Path
from subprocess import CalledProcessError


# Through sys.stderr rather than the file descriptor, so a progress display can capture it.
async def relay_stderr(stream: StreamReader) -> None:
    async for line in stream:
        sys.stderr.write(line.decode(errors="replace"))


async def subprocess_output(*command: str | Path) -> str:
    process = await asyncio.create_subprocess_exec(*command, stdout=PIPE, stderr=PIPE)
    stdout, _ = await asyncio.gather(process.stdout.read(), relay_stderr(process.stderr))
    if await process.wait():
        raise CalledProcessError(process.returncode, [str(part) for part in command])
    return stdout.decode().strip()


async def subprocess_check(*command: str | Path) -> bool:
    process = await asyncio.create_subprocess_exec(*command, stdout=DEVNULL, stderr=DEVNULL)
    return await process.wait() == 0
