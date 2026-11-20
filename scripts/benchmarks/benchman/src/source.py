import hashlib
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from benchman.nar import nar, source_files
from benchman.subprocess import subprocess_check, subprocess_output


def nix32(data: bytes) -> str:
    number = int.from_bytes(data, "little")
    digits = []
    for _ in range((len(data) * 8 + 4) // 5):
        number, digit = divmod(number, 32)
        digits.append("0123456789abcdfghijklmnpqrsvwxyz"[digit])
    return "".join(reversed(digits))


def pack(root: Path) -> tuple[str, bytes]:
    archive = nar(root, source_files(root))
    return hashlib.sha256(archive).hexdigest(), archive


def store_path(nar_hash: str) -> Path:
    fingerprint = f"source:sha256:{nar_hash}:/nix/store:source"
    folded = bytearray(20)
    for index, byte in enumerate(hashlib.sha256(fingerprint.encode()).digest()):
        folded[index % 20] ^= byte
    return Path("/nix/store", f"{nix32(folded)}-source")


@asynccontextmanager
async def unpack(nar_hash: str, download: Callable[[], Awaitable[bytes]]) -> AsyncIterator[Path]:
    path = store_path(nar_hash)
    with TemporaryDirectory() as temporary:
        directory = Path(temporary).resolve()
        root = directory / "root"
        if not await subprocess_check("nix-store", "--realise", "--add-root", root, path):
            archive = await download()
            (directory / "nix-cache-info").write_text("StoreDir: /nix/store\n")
            (directory / "source.nar").write_bytes(archive)
            (directory / f"{path.name.partition('-')[0]}.narinfo").write_text(
                f"StorePath: {path}\n"
                "URL: source.nar\n"
                "Compression: none\n"
                f"NarHash: sha256:{nar_hash}\n"
                f"NarSize: {len(archive)}\n"
                "References: \n"
                f"CA: fixed:r:sha256:{nar_hash}\n"
            )
            await subprocess_output("nix", "copy", "--from", f"file://{directory}", "--out-link", root, path)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield path
    finally:
        os.close(fd)
