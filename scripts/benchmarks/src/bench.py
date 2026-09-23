from enum import StrEnum

from pydantic import BaseModel, Field, JsonValue


class ChatRole(StrEnum):
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_call_id: str | None = Field(default=None, exclude_if=lambda value: value is None)


class BenchSampling(BaseModel):
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    temp: float | None = None


class BenchRequest(BaseModel):
    prompt_text: str | None = None
    prompt_chat: list[ChatMessage] | None = None
    tools: list[JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_choice: JsonValue | None = Field(default=None, exclude_if=lambda value: value is None)

    max_tokens: int | None = None
    speculative_depth: int | None = None
    sampling: BenchSampling | None = None
    num_runs: int | None = None

    @property
    def prompt(self) -> str | list[ChatMessage]:
        if self.prompt_text is not None:
            return self.prompt_text
        elif self.prompt_chat is not None:
            return self.prompt_chat
        else:
            raise ValueError("prompt_text and prompt_chat are None")


class BenchResponse(BaseModel):
    text: str
    time_to_first_token: float
    prompt_tps: float
    decode_tps: float
    tokens_per_forward_pass: float
    duration: float
    memory_phys_footprint: int
    memory_resident_peak: int
    memory_graphics_total: int
