"""Launch the ML Drift adapter, which builds the patched runner through run.sh."""

from pathlib import Path

import pytest

from .engine_tests import EngineTests

MODEL = "Qwen/Qwen3-0.6B"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("ML Drift")
    group.addoption("--mldrift-model", default=MODEL, help="Hugging Face checkpoint of an ML Drift model preset.")


class TestMLDrift(EngineTests):
    engine_name = "mldrift"
    supports_seed = False

    @staticmethod
    def command(config: pytest.Config, model: Path) -> list[str]:
        return ["./engine-mldrift/run.sh", str(model)]
