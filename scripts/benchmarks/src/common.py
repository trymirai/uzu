from pathlib import Path


def get_model_path(path: str | Path) -> str:
    if isinstance(path, Path):
        return str(path.expanduser())
    else:
        return path
