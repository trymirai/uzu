import subprocess
import uuid
from pathlib import Path

import pytest
from benchman.source import nix32, pack, unpack
from hypothesis import given
from hypothesis import strategies as st
from store import assert_alive, delete, git_tree


@pytest.mark.anyio
async def test_unpacks_into_store_once(tmp_path: Path) -> None:
    root = git_tree(tmp_path, {".gitignore": "ignored\n", "ignored": "", "unique": str(uuid.uuid4())})
    nar_hash, archive = pack(root)
    downloads = []

    async def download() -> bytes:
        downloads.append(nar_hash)
        return archive

    async with unpack(nar_hash, download) as path:
        assert sorted(child.name for child in path.iterdir()) == [".gitignore", "unique"]
        assert (path / "unique").read_text() == (root / "unique").read_text()
        async with unpack(nar_hash, download) as again:
            assert again == path
        assert downloads == [nar_hash]
        await assert_alive(path)
    await delete(path)


@given(st.binary(min_size=32, max_size=32))
def test_nix32_matches_nix(data: bytes) -> None:
    command = ["nix", "hash", "convert", "--hash-algo", "sha256", "--from", "base16", "--to", "nix32", data.hex()]
    assert nix32(data) == subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
