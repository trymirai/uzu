import asyncio
import subprocess
from asyncio.subprocess import PIPE
from pathlib import Path

from benchman.subprocess import subprocess_output


def git_tree(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(exist_ok=True)
    for name, content in files.items():
        (root / name).write_text(content)
    subprocess.run(["git", "init", "--quiet", root], check=True)
    return root


async def assert_alive(path: Path) -> None:
    process = await asyncio.create_subprocess_exec("nix", "store", "delete", path, stdout=PIPE, stderr=PIPE)
    assert b"since it is still alive" in (await process.communicate())[1]


async def delete(path: Path) -> None:
    await subprocess_output("nix", "store", "delete", path)
