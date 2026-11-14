from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from prefect import Task
from prefect.cache_policies import NO_CACHE
from prefect.task_engine import AsyncTaskRunEngine

T = TypeVar("T")


class TaskBlock:
    output: Any = None

    def set_output(self, output: T) -> T:
        self.output = output
        return output


async def block() -> None:
    pass


@asynccontextmanager
async def task(name: str, *upstream: Any) -> AsyncIterator[TaskBlock]:
    engine = AsyncTaskRunEngine(
        task=Task(block, name=name, task_run_name=name, cache_policy=NO_CACHE, persist_result=False),
        wait_for=list(upstream),
    )
    async with engine.start(), engine.run_context(), engine.transaction_context() as transaction:
        body = TaskBlock()
        yield body
        await engine.handle_success(body.output, transaction)
    await engine.result()
