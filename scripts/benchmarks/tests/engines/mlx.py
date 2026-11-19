"""Launch the MLX benchmark adapter in its own uv project."""

import pytest

from .engine_tests import EngineTests

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("MLX")
    group.addoption("--mlx-model", default=MODEL, help="Target model local path or HuggingFace id.")


class TestMLX(EngineTests):
    engine_name = "mlx"
