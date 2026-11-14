from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_call_id: str | None = Field(default=None, exclude_if=lambda value: value is None)


class GreedySampling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    greedy: Literal[True]


class StochasticSampling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int | None
    top_p: float | None
    min_p: float | None
    temp: float | None
    repetition_penalty: float | None
    suffix_repetition_length: int | None


BenchSampling = GreedySampling | StochasticSampling

ReasoningEffort = Literal["disabled", "default", "low", "medium", "high", "xhigh"]
REASONING_LEVELS: tuple[ReasoningEffort, ...] = ("low", "medium", "high", "xhigh")


class ReasoningEfforts(BaseModel):
    """For each effort a caller may want, the effort to request from this model, which it supports."""

    model_config = ConfigDict(extra="forbid")
    disabled: ReasoningEffort
    default: ReasoningEffort
    low: ReasoningEffort
    medium: ReasoningEffort
    high: ReasoningEffort
    xhigh: ReasoningEffort


class ModelConfig(BaseModel):
    """What every engine's model.json has; engines extend it with their own fields."""

    model_config = ConfigDict(extra="forbid")
    model: str
    specdec: bool
    sampling: BenchSampling
    reasoning_efforts: ReasoningEfforts


class BenchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["BenchRequest"] = "BenchRequest"
    messages: list[ChatMessage]
    tools: list[JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_choice: JsonValue | None = Field(default=None, exclude_if=lambda value: value is None)

    max_output_tokens: int | None = Field(default=None, ge=0)
    reasoning_effort: ReasoningEffort
    seed: int | None = Field(default=None, ge=0, lt=2**32)


class CooldownRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["CooldownRequest"] = "CooldownRequest"
    temperature: float = 60.0
    only_if_capped: bool = False


Request = Annotated[BenchRequest | CooldownRequest, Field(discriminator="type")]


class BenchResponse(BaseModel):
    type: Literal["BenchResponse"] = "BenchResponse"
    text: str
    finish_reason: Literal["stop", "length"]
    input_tokens_count: int
    output_tokens_count: int
    time_to_first_token: float
    prefill_tps: float
    decode_tps: float
    tokens_per_forward_pass: float | None
    duration: float
    energy_gpu: float
    memory_used: int
    memory_phys_footprint: int
    memory_resident: int
    memory_graphics_total: int


class ReadyResponse(BaseModel):
    type: Literal["ReadyResponse"] = "ReadyResponse"


class CooldownResponse(BaseModel):
    type: Literal["CooldownResponse"] = "CooldownResponse"
