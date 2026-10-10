"""Launch the ML Drift adapter, which builds the patched runner through run.sh."""

import pytest

from .engine_process import EngineProcess
from .engine_tests import EngineTests

MODEL = "Qwen/Qwen3-0.6B"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("ML Drift")
    group.addoption("--mldrift-model", default=MODEL, help="Hugging Face checkpoint of an ML Drift model preset.")


class TestMLDrift(EngineTests):
    engine_name = "mldrift"

    @staticmethod
    def command(config: pytest.Config) -> list[str]:
        return ["./engine-mldrift/run.sh", "--model", config.getoption("mldrift_model")]

    def test_sampling(self, engine: EngineProcess) -> None:
        pytest.skip("ML Drift decodes greedily on the GPU and rejects sampling")
