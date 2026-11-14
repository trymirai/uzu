import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import zstandard
from benchman.source import pack, unpack
from prefect.filesystems import WritableFileSystem

PREFIX = "uzu-bench"


def nar_key(nar_hash: str) -> str:
    return f"{PREFIX}/nar/{nar_hash}.nar.zst"


async def upload_nar(root: Path, storage: WritableFileSystem) -> str:
    nar_hash, archive = await asyncio.to_thread(pack, root)
    compressed = await asyncio.to_thread(zstandard.ZstdCompressor(level=10).compress, archive)
    await storage.write_path(nar_key(nar_hash), compressed)
    return nar_hash


async def download(storage: WritableFileSystem, nar_hash: str) -> bytes:
    return zstandard.ZstdDecompressor().decompress(await storage.read_path(nar_key(nar_hash)))


def download_nar(nar_hash: str, storage: WritableFileSystem) -> AbstractAsyncContextManager[Path]:
    return unpack(nar_hash, lambda: download(storage, nar_hash))
