from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class ChatRole(StrEnum):
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


@dataclass
class ChatMessage:
    message: str
    role: ChatRole


class BenchSampling(BaseModel):
    top_k: int | None
    top_p: float | None
    min_p: float | None
    temp: float | None


class BenchInput(BaseModel):
    prompt_text: str | None
    prompt_chat: list[ChatMessage] | None
    max_tokens: int | None
    speculative_depth: int | None
    sampling: BenchSampling | None

    @property
    def prompt(self) -> str | list[ChatMessage]:
        if self.prompt_text is not None:
            return self.prompt_text
        elif self.prompt_chat is not None:
            return self.prompt_chat
        else:
            raise ValueError("prompt_text and prompt_chat are None")


class BenchOutput(BaseModel):
    text: str
    time_to_first_token: float
    prompt_tps: float
    decode_tps: float
    tokens_per_forward_pass: float
    duration: float
    memory_phys_footprint: int
    memory_resident_peak: int
    memory_graphics_total: int


class Engine(StrEnum):
    LLAMA_CPP = "llamacpp"
    MLX = "mlx"
    MTPLX = "mtplx"


def get_model_path(path: str | Path) -> str:
    if isinstance(path, Path):
        return str(path.expanduser())
    else:
        return path
