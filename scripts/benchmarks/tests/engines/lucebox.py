"""Launch Lucebox's luce_server on CUDA: Linux only, inside WSL on Windows."""

import pytest
from wsl import linux_uv_command

from .engine_tests import EngineTests

MODEL = "Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF:Qwen3.8-27B-IQ4_XS-pure.gguf"
DRAFT_MODEL = "Lucebox/Qwen3.8-27B-DFlash2-GGUF:Qwen3.8-27B-DFlash2-Q8_0.gguf"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("Lucebox")
    group.addoption("--lucebox-model", default=MODEL, help="Target GGUF: a path WSL can open or REPOSITORY:FILE.gguf.")
    group.addoption("--lucebox-draft-model", default=DRAFT_MODEL, help="DFlash draft GGUF, in the same forms.")


class TestLucebox(EngineTests):
    engine_name = "lucebox"
    chat_only = True

    @staticmethod
    def command(config: pytest.Config) -> list[str]:
        args = ["--model", config.getoption("lucebox_model"), "--draft-model", config.getoption("lucebox_draft_model")]
        return linux_uv_command("engine-lucebox", "bench-lucebox", args)
