from benchman.device import Hardware
from prefect.client.orchestration import get_client
from prefect.client.schemas.objects import Worker, WorkerStatus
from pydantic import Field


def configured() -> bool:
    try:
        with get_client(sync_client=True):
            return True
    except ValueError:
        return False


def work_pools() -> list[str]:
    with get_client(sync_client=True) as client:
        return sorted(work_pool.name for work_pool in client.read_work_pools())


class TaggedWorker(Worker):
    """A worker with the tags that the fleet's server reports and the client drops."""

    tags: list[str] = Field(default_factory=list)


def queues(work_pool: str) -> dict[str, Hardware]:
    """The queues that online workers serve, from the workers' tags."""
    with get_client(sync_client=True) as client:
        path = "/work_pools/{work_pool_name}/workers/filter"
        response = client.request("POST", path, path_params={"work_pool_name": work_pool}, json={})
    served: dict[str, Hardware] = {}
    for worker in [TaggedWorker.model_validate(worker) for worker in response.json()]:
        if worker.status != WorkerStatus.ONLINE:
            continue
        tags = {
            kind: [tag.removeprefix(f"tag:{kind}-") for tag in worker.tags if tag.startswith(f"tag:{kind}-")]
            for kind in ("ram", "chip", "queue")
        }
        if len(tags["ram"]) != 1 or len(tags["chip"]) != 1 or not tags["queue"]:
            raise RuntimeError(
                f"Worker {worker.name} needs one tag:ram-<N>gb, one tag:chip-<chip> and a tag:queue-<name>"
            )
        memory = int(tags["ram"][0].removesuffix("gb")) * 2**30
        for name in tags["queue"]:
            queue = served.get(name, Hardware(memory, frozenset()))
            served[name] = Hardware(min(queue.memory, memory), queue.chips | {tags["chip"][0]})
    return served
