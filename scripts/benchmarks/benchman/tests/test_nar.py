import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchman.nar import nar, source_files
from hypothesis import given
from hypothesis import strategies as st

names = st.text("abcxyz019-_.", min_size=1, max_size=6).filter(lambda name: name not in (".", ".."))
leaves = st.tuples(st.binary(max_size=40), st.booleans()) | names.map(Path)
nodes = st.recursive(leaves, lambda children: st.dictionaries(names, children, min_size=1, max_size=4), max_leaves=16)
ignored = names.map(lambda name: f"{name}.ignored") | names.map(lambda name: f"{name}.global")
trees = st.dictionaries(names | ignored, nodes, max_size=6)


def write(tree: dict, directory: Path, *, ignored: bool) -> None:
    for name, node in tree.items():
        path = directory / name
        if name.endswith((".ignored", ".global")) and not ignored:
            continue
        match node:
            case dict():
                path.mkdir()
                write(node, path, ignored=ignored)
                if not any(path.iterdir()):
                    path.rmdir()
            case Path():
                path.symlink_to(node)
            case (contents, executable):
                path.write_bytes(contents)
                path.chmod(0o755 if executable else 0o644)


@given(trees)
def test_matches_nix_dump_of_non_ignored_files(tree: dict) -> None:
    with TemporaryDirectory() as temporary:
        source, expected = (Path(temporary).resolve() / name for name in ("source", "expected"))
        for root, ignored in ((source, True), (expected, False)):
            root.mkdir()
            (root / ".gitignore").write_text("*.ignored\n")
            write(tree, root, ignored=ignored)
        (Path(temporary) / "ignore").write_text("*.global\n")
        (Path(temporary) / "config").write_text(f"[core]\n\texcludesFile = {Path(temporary) / 'ignore'}\n")
        with patch.dict(os.environ, GIT_CONFIG_GLOBAL=str(Path(temporary) / "config"), GIT_CONFIG_NOSYSTEM="1"):
            subprocess.run(["git", "init", "--quiet", source], check=True)
            files = source_files(source)
        dump = subprocess.run(["nix-store", "--dump", expected], check=True, capture_output=True).stdout
        assert nar(source, files) == dump
