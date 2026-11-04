import os
import shutil
from pathlib import Path

PROFILE = Path("/nix/var/nix/profiles/default/bin")


def add_nix_to_path() -> None:
    if shutil.which("nix") is None and PROFILE.is_dir():
        os.environ["PATH"] = f"{PROFILE}:{os.environ['PATH']}"
