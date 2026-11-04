import asyncio
from asyncio.subprocess import DEVNULL, PIPE
from pathlib import Path
from subprocess import CalledProcessError


async def subprocess_output(*command: str | Path) -> str:
    process = await asyncio.create_subprocess_exec(*command, stdout=PIPE)
    stdout, _ = await process.communicate()
    if process.returncode:
        raise CalledProcessError(process.returncode, command)
    return stdout.decode().strip()


async def subprocess_check(*command: str | Path) -> bool:
    process = await asyncio.create_subprocess_exec(*command, stdout=DEVNULL, stderr=DEVNULL)
    return await process.wait() == 0
