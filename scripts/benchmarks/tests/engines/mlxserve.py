"""Launch the source-built native mlx-serve benchmark runner."""

from pathlib import Path

import pytest

from .engine_tests import EngineTests

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("mlx-serve")
    group.addoption("--mlxserve-model", default=MODEL, help="Target MLX model directory or HuggingFace id.")
    group.addoption("--mlxserve-draft-model", help="DFlash or Gemma assistant directory or HuggingFace id.")


class TestMLXServe(EngineTests):
    engine_name = "mlxserve"

    @staticmethod
    def command(config: pytest.Config, model: Path) -> list[str]:
        return ["./engine-mlxserve/run.sh", str(model)]
