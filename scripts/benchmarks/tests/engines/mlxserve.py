"""Launch the source-built native mlx-serve benchmark runner."""

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
    def command(config: pytest.Config) -> list[str]:
        args = ["./engine-mlxserve/run.sh", "--model", config.getoption("mlxserve_model")]
        if draft := config.getoption("mlxserve_draft_model"):
            args.extend(["--draft-model", draft])
        return args
