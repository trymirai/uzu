import uuid
from pathlib import Path

import pytest
from benchman.storage import download_nar, nar_key, upload_nar
from prefect.filesystems import LocalFileSystem
from store import assert_alive, delete, git_tree


@pytest.mark.anyio
async def test_downloads_only_when_missing(tmp_path: Path) -> None:
    root = git_tree(tmp_path / "source", {"unique": str(uuid.uuid4())})
    storage = LocalFileSystem(basepath=str(tmp_path / "storage"))
    nar_hash = await upload_nar(root, storage)

    async with download_nar(nar_hash, storage) as path:
        assert (path / "unique").read_text() == (root / "unique").read_text()
        (tmp_path / "storage" / nar_key(nar_hash)).unlink()
        async with download_nar(nar_hash, storage) as again:
            assert again == path
        await assert_alive(path)
    await delete(path)
