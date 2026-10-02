"""Launch Splash with the smallest supported target GGUF variant."""

import pytest

MODEL = "unsloth/Qwen3.8-27B-GGUF:UD-IQ1_S"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("Splash")
    group.addoption(
        "--splash-model", default=MODEL, help="Prepared model local path or HuggingFace id (with optional :VARIANT)."
    )


def command(config: pytest.Config) -> list[str]:
    return ["uv", "run", "splash", "--model", config.getoption("splash_model")]
