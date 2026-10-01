"""Launch the MLX benchmark adapter through uv run mlx."""

import pytest

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("MLX")
    group.addoption("--mlx-model", default=MODEL, help="Target model local path or HuggingFace id.")


def command(config: pytest.Config) -> list[str]:
    return ["uv", "run", "mlx", "--model", config.getoption("mlx_model")]
