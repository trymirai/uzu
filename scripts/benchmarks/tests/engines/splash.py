"""Launch Splash with the smallest supported target GGUF variant."""

from pathlib import Path

import pytest

from .engine_tests import EngineTests

MODEL = "unsloth/Qwen3.8-27B-GGUF:UD-IQ1_S"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("Splash")
    group.addoption(
        "--splash-model", default=MODEL, help="Prepared model local path or HuggingFace id (with optional :VARIANT)."
    )


class TestSplash(EngineTests):
    engine_name = "splash"
    specdec = True

    @staticmethod
    def command(config: pytest.Config, model: Path) -> list[str]:
        return ["uv", "run", "--project", "engine-splash", "bench-splash", str(model)]
