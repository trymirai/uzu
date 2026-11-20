"""Launch Kapitsa with a small MLX model and its DFlash draft."""

import pytest

from .engine_tests import EngineTests

MODEL = "lmstudio-community/Qwen3.5-4B-MLX-4bit"
DRAFT_MODEL = "z-lab/Qwen3.5-4B-DFlash"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("Kapitsa")
    group.addoption("--kapitsa-model", default=MODEL, help="Target model local path or HuggingFace id.")
    group.addoption("--kapitsa-draft-model", default=DRAFT_MODEL, help="Draft model local path or HuggingFace id.")


class TestKapitsa(EngineTests):
    engine_name = "kapitsa"
