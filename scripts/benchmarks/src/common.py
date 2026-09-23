from enum import StrEnum
from pathlib import Path


class Engine(StrEnum):
    LLAMA_CPP = "llamacpp"
    MLX = "mlx"
    MTPLX = "mtplx"


def get_model_path(path: str | Path) -> str:
    if isinstance(path, Path):
        return str(path.expanduser())
    else:
        return path
