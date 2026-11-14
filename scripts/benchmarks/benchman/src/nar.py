import os
import stat
import subprocess
from pathlib import Path


def source_files(root: Path) -> set[Path]:
    output = subprocess.run(
        ["git", "-C", root, "ls-files", "-coz", "--exclude-standard"], check=True, capture_output=True
    )
    paths = (root / os.fsdecode(name) for name in output.stdout.split(b"\0") if name)
    return {path for path in paths if os.path.lexists(path)}


def nar(root: Path, files: set[Path]) -> bytes:
    kept = files | {parent for file in files for parent in file.parents}
    archive = bytearray()

    def strings(*values: bytes) -> None:
        for value in values:
            archive.extend(len(value).to_bytes(8, "little") + value + bytes(-len(value) % 8))

    def node(path: Path) -> None:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            strings(b"(", b"type", b"symlink", b"target", os.fsencode(os.readlink(path)))
        elif stat.S_ISDIR(mode) and path not in files:
            strings(b"(", b"type", b"directory")
            for child in sorted(kept.intersection(path.iterdir()), key=lambda child: os.fsencode(child.name)):
                strings(b"entry", b"(", b"name", os.fsencode(child.name), b"node")
                node(child)
                strings(b")")
        else:
            executable = (b"executable", b"") if mode & stat.S_IXUSR else ()
            strings(b"(", b"type", b"regular", *executable, b"contents", path.read_bytes())
        strings(b")")

    strings(b"nix-archive-1")
    node(root)
    return bytes(archive)
