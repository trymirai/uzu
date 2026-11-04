from contextlib import AsyncExitStack
from typing import Any

from bench import BenchRequest, BenchResponse, ChatMessage, CooldownRequest
from benchman.engine import build_engine, cool_down, run_engine, run_task
from benchman.nix_path_hack import add_nix_to_path
from benchman.storage import download_nar
from benchman.task import task
from prefect import flow
from prefect.artifacts import acreate_markdown_artifact
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import ArtifactCreate
from prefect.context import FlowRunContext
from prefect.utilities.context import get_task_and_flow_run_ids
from pydantic import BaseModel, Field


class TaskRequest(BenchRequest):
    repeats: int = Field(default=1, ge=1)

    def bench_request(self) -> BenchRequest:
        return BenchRequest.model_validate(self.model_dump(exclude={"repeats"}))


class RunResult(BaseModel):
    task: str
    repeat: int
    response: BenchResponse


# The Prefect UI's run graph only renders its built-in artifact types.
RUN_RESULT_ARTIFACT_TYPE = "unknown"


@flow
async def benchmark(nar_hash: str, engine: str, config: dict[str, Any], tasks: dict[str, TaskRequest]) -> None:
    add_nix_to_path()
    storage = FlowRunContext.get().result_store.result_storage
    async with AsyncExitStack() as stack:
        async with task("download archive") as block:
            source = block.set_output(await stack.enter_async_context(download_nar(nar_hash, storage)))
        async with task(f"build {engine} engine", source) as block:
            executable = block.set_output(await stack.enter_async_context(build_engine(source, engine)))
        async with task(f"load {config['model']}", executable) as block:
            process = block.set_output(await stack.enter_async_context(run_engine(executable, config)))
        async with task("warmup", process) as block:
            warmup = BenchRequest(
                messages=[ChatMessage(role="user", content="warmup")], max_output_tokens=1, reasoning_effort="default"
            )
            previous = block.set_output(await run_task(process, warmup))
        for name, request in tasks.items():
            for repeat in range(request.repeats):
                async with task("cooldown", previous) as block:
                    cooled = block.set_output(await cool_down(process, CooldownRequest(only_if_capped=True)))
                label = f"run {name}" + (f" {repeat + 1}/{request.repeats}" if request.repeats > 1 else "")
                async with task(label, cooled) as block:
                    response = await run_task(process, request.bench_request())
                    task_run_id, flow_run_id = get_task_and_flow_run_ids()
                    async with get_client() as client:
                        await client.create_artifact(
                            ArtifactCreate(
                                type=RUN_RESULT_ARTIFACT_TYPE,
                                description=name,
                                data=RunResult(task=name, repeat=repeat, response=response).model_dump(mode="json"),
                                task_run_id=task_run_id,
                                flow_run_id=flow_run_id,
                            )
                        )
                    summary = f"- Decode: {response.decode_tps:.1f} tokens/s\n"
                    if response.tokens_per_forward_pass is not None:
                        summary += f"- Mean accepted length: {response.tokens_per_forward_pass:.2f}\n"
                    summary += (
                        f"- Prefill: {response.prefill_tps:.0f} tokens/s\n"
                        f"- Time to first token: {response.time_to_first_token:.2f} s\n"
                        f"- Memory: {response.memory_used / 2**30:.2f} GiB\n"
                    )
                    await acreate_markdown_artifact(markdown=summary, description=name)
                    previous = block.set_output(response)
