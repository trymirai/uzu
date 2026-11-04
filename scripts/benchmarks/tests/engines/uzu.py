"""Launch the Uzu benchmark adapter with an optimized Cargo build."""

from pathlib import Path

import pytest

from .engine_tests import EngineTests

MODEL = "trymirai/LFM2.5-230M-M"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("Uzu")
    group.addoption("--uzu-model", default=MODEL, help="Target Uzu model local path or HuggingFace id.")


class TestUzu(EngineTests):
    engine_name = "uzu"

    @staticmethod
    def command(config: pytest.Config, model: Path) -> list[str]:
        return ["cargo", "run", "--release", "-p", "benchmarks-uzu", "--", str(model)]
