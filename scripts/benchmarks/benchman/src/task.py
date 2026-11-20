from collections.abc import Iterable
from statistics import fmean
from typing import Any, Self
from uuid import UUID

from bench import BenchRequest, BenchResponse, ReadyResponse
from benchman.device import Device
from pydantic import BaseModel, Field
from reasoning import map_reasoning_efforts


class TaskRequest(BenchRequest):
    warmups: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    repeats: int = Field(default=1, ge=1)
    specdec_filter: bool | None = None

    def bench_request(self) -> BenchRequest:
        return BenchRequest.model_validate(self.model_dump(exclude={"warmups", "repeats", "specdec_filter"}))

    def warmup_step(self, name: str, warmup: int) -> str:
        return f"warmup {name}" + (f" {warmup + 1}/{self.warmups}" if self.warmups > 1 else "")

    def step(self, name: str, repeat: int) -> str:
        return f"run {name}" + (f" {repeat + 1}/{self.repeats}" if self.repeats > 1 else "")


def resolve_tasks(
    tasks: dict[str, TaskRequest], config: dict[str, Any], ready: ReadyResponse
) -> dict[str, TaskRequest]:
    efforts = config.get("reasoning_efforts") or map_reasoning_efforts(ready.supported_reasoning_efforts)
    return {
        name: task.model_copy(update={"reasoning_effort": efforts[task.reasoning_effort]})
        for name, task in tasks.items()
        if task.specdec_filter in (None, ready.specdec)
    }


def mean_of_present(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


class Summary(BaseModel):
    input_tokens_count: float
    output_tokens_count: float
    time_to_first_token: float
    prefill_tps: float
    decode_tps: float
    tokens_per_forward_pass: float | None
    forwards_per_second: float | None
    duration: float
    energy_gpu: float
    memory_used: int
    memory_phys_footprint: int
    memory_resident: int
    memory_graphics_total: int


def summarize(parts: Iterable[BenchResponse | Summary]) -> Summary:
    """The means over a task's repeats or over tasks, and the highest memory use."""
    parts = list(parts)
    return Summary(
        input_tokens_count=fmean(part.input_tokens_count for part in parts),
        output_tokens_count=fmean(part.output_tokens_count for part in parts),
        time_to_first_token=fmean(part.time_to_first_token for part in parts),
        prefill_tps=fmean(part.prefill_tps for part in parts),
        decode_tps=fmean(part.decode_tps for part in parts),
        tokens_per_forward_pass=mean_of_present(part.tokens_per_forward_pass for part in parts),
        forwards_per_second=mean_of_present(part.forwards_per_second for part in parts),
        duration=fmean(part.duration for part in parts),
        energy_gpu=fmean(part.energy_gpu for part in parts),
        memory_used=max(part.memory_used for part in parts),
        memory_phys_footprint=max(part.memory_phys_footprint for part in parts),
        memory_resident=max(part.memory_resident for part in parts),
        memory_graphics_total=max(part.memory_graphics_total for part in parts),
    )


class TaskResponse(Summary):
    responses: list[BenchResponse]

    @classmethod
    def aggregate(cls, responses: list[BenchResponse]) -> Self:
        input_tokens = {response.input_tokens_count for response in responses}
        if len(input_tokens) != 1:
            raise ValueError(f"Repeats disagree on input tokens: {sorted(input_tokens)}")
        accepted = [response.tokens_per_forward_pass for response in responses]
        if None in accepted and any(accepted):
            raise ValueError("Speculation was reported for only some repeats")
        return cls(responses=responses, **summarize(responses).model_dump())


class Failure(BaseModel):
    message: str


class Outcome(BaseModel):
    device: Device
    # None if the engine failed before it was ready.
    ready: ReadyResponse | None
    # None if the engine doesn't decode the way the task is for.
    result: TaskResponse | Failure | None


class TaskRecord(BaseModel):
    """The flow run that has a task to measure, and once it has, how that ended."""

    flow_run_id: UUID
    # The tags that get a copy of the outcome.
    tags: list[str] = Field(default_factory=list)
    outcome: Outcome | None = None
    # Of the stored object that was read: what a change of it goes over.
    generation: int = Field(default=0, exclude=True)
