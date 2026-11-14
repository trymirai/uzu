"""Launch oMLX with a small instruction model supporting text, chat, and EOS."""

from pathlib import Path

import pytest

from .engine_tests import EngineTests

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("oMLX")
    group.addoption("--omlx-model", default=MODEL, help="Target model local path or HuggingFace id.")
    group.addoption("--omlx-draft-model", help="DFlash draft model local path or HuggingFace id.")


class TestOMLX(EngineTests):
    engine_name = "omlx"

    @staticmethod
    def command(config: pytest.Config, model: Path) -> list[str]:
        return ["uv", "run", "--project", "engine-omlx", "bench-omlx", str(model)]
