"""Launch vLLM on CUDA: Linux only, inside WSL on Windows."""

import pytest
from wsl import linux_uv_command

from .engine_tests import EngineTests

MODEL = "Qwen/Qwen3-0.6B"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("vLLM")
    group.addoption("--vllm-model", default=MODEL, help="Target model Hugging Face id or path WSL can open.")
    group.addoption("--vllm-draft-model", default=None, help="DFlash draft model for speculative decoding.")


class TestVLLM(EngineTests):
    engine_name = "vllm"

    @staticmethod
    def command(config: pytest.Config) -> list[str]:
        args = ["--model", config.getoption("vllm_model")]
        if (draft := config.getoption("vllm_draft_model")) is not None:
            args += ["--draft-model", draft]
        return linux_uv_command("engine-vllm", "bench-vllm", args)
