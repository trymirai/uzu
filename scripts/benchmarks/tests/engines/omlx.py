"""Launch oMLX with a small instruction model supporting text, chat, and EOS."""

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
    def command(config: pytest.Config) -> list[str]:
        args = ["uv", "run", "--project", "engine-omlx", "bench-omlx", "--model", config.getoption("omlx_model")]
        if draft := config.getoption("omlx_draft_model"):
            args.extend(["--draft-model", draft])
        return args
