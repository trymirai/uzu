"""Launch MTPLX with its smallest published native-MTP model pack."""

import pytest

MODEL = "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("MTPLX")
    group.addoption(
        "--mtplx-model", default=MODEL, help="Target model local path or HuggingFace id (with MTP weights)."
    )


def command(config: pytest.Config) -> list[str]:
    return ["uv", "run", "mtplx", "--model", config.getoption("mtplx_model")]
