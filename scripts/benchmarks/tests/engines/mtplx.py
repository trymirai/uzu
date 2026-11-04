"""Launch MTPLX with its smallest published native-MTP model pack."""

from pathlib import Path

import pytest

from .engine_tests import EngineTests

MODEL = "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("MTPLX")
    group.addoption(
        "--mtplx-model", default=MODEL, help="Target model local path or HuggingFace id (with MTP weights)."
    )


class TestMTPLX(EngineTests):
    engine_name = "mtplx"
    specdec = True

    @staticmethod
    def command(config: pytest.Config, model: Path) -> list[str]:
        return ["uv", "run", "--project", "engine-mtplx", "bench-mtplx", str(model)]
