"""Launch the llama.cpp benchmark adapter, or an already-built one."""

import os
from pathlib import Path

import pytest

from .engine_tests import EngineTests

MODEL = "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_S"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("llama.cpp")
    group.addoption("--llamacpp-binary", type=Path, help="Use an already-built engine-llamacpp instead of building.")
    group.addoption("--llamacpp-model", default=MODEL, help="Target model local path or HuggingFace id.")
    group.addoption("--llamacpp-draft-model", help="Matching DFlash model; also enables DFlash regression tests.")


class TestLlamaCpp(EngineTests):
    engine_name = "llamacpp"

    @classmethod
    def command(cls, config: pytest.Config, model: Path) -> list[str]:
        binary = config.getoption("llamacpp_binary")
        if binary is None:
            return super().command(config, model)
        binary = binary.expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise pytest.UsageError(f"--llamacpp-binary is not an executable file: {binary}")
        return [str(binary), str(model)]
