import pyarrow.parquet as pq
from bench import BenchRequest, ChatMessage
from benchman.datasets.dataset import Dataset, download_dataset_file


async def load() -> Dataset:
    path = await download_dataset_file(
        "openai/openai_humaneval",
        "openai_humaneval/test-00000-of-00001.parquet",
        revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    )
    return {
        row["task_id"]: BenchRequest(
            messages=[ChatMessage(role="user", content=row["prompt"])], reasoning_effort="default"
        )
        for row in pq.read_table(path).to_pylist()
    }
