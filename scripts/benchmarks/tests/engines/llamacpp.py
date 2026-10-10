"""Launch the benchmark adapter, which builds llama.cpp through run.sh."""

import os
from pathlib import Path

import pytest

from .engine_tests import EngineTests

ROOT = Path(__file__).resolve().parents[2]
MODEL = "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("llama.cpp")
    group.addoption("--llamacpp-binary", type=Path, help="Use an already-built engine-llamacpp instead of run.sh.")
    group.addoption("--llamacpp-model", default=MODEL, help="Target model local path or HuggingFace id.")
    group.addoption("--llamacpp-draft-model", help="Matching DFlash model; also enables DFlash regression tests.")


class TestLlamaCpp(EngineTests):
    engine_name = "llamacpp"

    @staticmethod
    def command(config: pytest.Config) -> list[str]:
        args = ["--model", config.getoption("llamacpp_model")]
        if draft_model := config.getoption("llamacpp_draft_model"):
            args.extend(["--draft-model", draft_model])
        binary = config.getoption("llamacpp_binary")
        if binary is not None:
            binary = binary.expanduser().resolve()
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise pytest.UsageError(f"--llamacpp-binary is not an executable file: {binary}")
            return [str(binary), *args]
        return [str(ROOT / "engine-llamacpp/run.sh"), *args]
