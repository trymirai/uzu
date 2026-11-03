#!/usr/bin/env python3
"""Move Mirai data aside for a fresh-start preview, then restore over test data."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


PATHS = {
    "app": "Library/Application Support/mirai",
    "cache": ".cache/mirai",
    "webview": "Library/WebKit/com.mirai.tech.chat",
    "webview-cache": "Library/Caches/com.mirai.tech.chat",
    "tauri": "Library/Application Support/com.mirai.tech.chat",
    "preferences": "Library/Preferences/com.mirai.tech.chat.plist",
    "window-state": "Library/Saved Application State/com.mirai.tech.chat.savedState",
}
MANIFEST = "manifest.json"


def exists(path):
    return os.path.lexists(path)


def check_location(data_home, backup_dir):
    for relative in PATHS.values():
        source = (data_home / relative).resolve()
        if source == backup_dir or source in backup_dir.parents or backup_dir in source.parents:
            raise ValueError(f"Backup directory overlaps Mirai data: {source}")


def check_filesystem(paths, directory):
    device = directory.stat().st_dev
    for path in paths:
        if exists(path) and path.lstat().st_dev != device:
            raise ValueError(f"Choose a backup directory on the same filesystem as {path}")


def move(source, destination, moved):
    if exists(destination):
        raise ValueError(f"Move destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    moved.append((source, destination))


def rollback(moved):
    for source, destination in reversed(moved):
        if exists(source):
            raise ValueError(f"Cannot roll back over {source}; recovery data remains at {destination}")
        source.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(source)


def backup(data_home, backup_dir, keep_cache=False):
    check_location(data_home, backup_dir)
    entries = {
        name: exists(data_home / relative)
        for name, relative in PATHS.items()
        if not (keep_cache and name == "cache")
    }
    backup_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    moved = []
    try:
        check_filesystem((data_home / PATHS[name] for name in entries), backup_dir)
        for name, present in entries.items():
            if present:
                move(data_home / PATHS[name], backup_dir / name, moved)
        (backup_dir / MANIFEST).write_text(json.dumps({
            "version": 1,
            "home": str(data_home),
            "entries": entries,
        }, indent=2) + "\n")
    except BaseException:
        rollback(moved)
        (backup_dir / MANIFEST).unlink(missing_ok=True)
        backup_dir.rmdir()
        raise
    print(f"Moved Mirai data to {backup_dir}")
    if keep_cache:
        print("Left ~/.cache/mirai in place; restore will also leave it untouched.")


def restore(data_home, backup_dir):
    check_location(data_home, backup_dir)
    manifest = json.loads((backup_dir / MANIFEST).read_text())
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("home") != str(data_home):
        raise ValueError("Backup version or home directory does not match")
    entries = manifest.get("entries")
    if (
        not isinstance(entries, dict)
        or set(entries) not in (set(PATHS), set(PATHS) - {"cache"})
        or any(type(value) is not bool for value in entries.values())
    ):
        raise ValueError("Invalid backup manifest")
    expected = {MANIFEST} | {name for name, present in entries.items() if present}
    if {path.name for path in backup_dir.iterdir()} != expected:
        raise ValueError("Backup contents do not match the manifest; no current data was changed")

    staging = Path(tempfile.mkdtemp(prefix=".mirai-restore-", dir=backup_dir.parent))
    moved = []
    try:
        check_filesystem((data_home / PATHS[name] for name in entries), staging)
        check_filesystem((backup_dir / name for name, present in entries.items() if present), staging)
        for name in entries:
            current = data_home / PATHS[name]
            if exists(current):
                move(current, staging / name, moved)
        for name, present in entries.items():
            if present:
                move(backup_dir / name, data_home / PATHS[name], moved)
        move(backup_dir / MANIFEST, staging / MANIFEST, moved)
        backup_dir.rmdir()
    except BaseException:
        rollback(moved)
        staging.rmdir()
        raise
    shutil.rmtree(staging)
    print("Restored Mirai data, replacing the test data. Backup consumed.")
    if "cache" not in entries:
        print("Left the current ~/.cache/mirai untouched.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Quit Mirai, the CLI, and other apps using its shared cache before either operation. "
        "Use a new backup directory on the same filesystem as the data. "
        "restore-overwrite discards current test data and consumes the backup.",
    )
    parser.add_argument("action", choices=("backup", "restore-overwrite"))
    parser.add_argument("backup_dir", type=Path)
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="backup only: leave ~/.cache/mirai in place; restore preserves it too",
    )
    args = parser.parse_args()
    if args.keep_cache and args.action != "backup":
        parser.error("--keep-cache is only for backup; restore follows the saved manifest")
    data_home = Path.home().resolve()
    backup_dir = args.backup_dir.expanduser().resolve()
    try:
        running = subprocess.run(
            ["pgrep", "-x", "Mirai|mirai-gui|mirai|cli"], capture_output=True, text=True,
        )
        if running.returncode == 0:
            raise ValueError(f"Quit Mirai and the CLI first (PIDs: {', '.join(running.stdout.split())})")
        if running.returncode != 1:
            raise ValueError(f"Cannot check for running Mirai processes: {running.stderr.strip()}")
        if args.action == "backup":
            backup(data_home, backup_dir, args.keep_cache)
        else:
            restore(data_home, backup_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
