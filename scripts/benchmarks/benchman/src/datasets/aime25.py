import pyarrow.parquet as pq
from bench import BenchRequest, ChatMessage
from benchman.datasets.dataset import Dataset, download_dataset_file


async def load() -> Dataset:
    path = await download_dataset_file(
        "test-time-compute/aime_2025",
        "data/test-00000-of-00001.parquet",
        revision="a7b26602f8321cf71132eeb57c7f4d163e9b5a50",
    )
    return {
        str(row["metadata"]["problem_idx"]): BenchRequest(
            messages=[ChatMessage(role="user", content=row["question"])], reasoning_effort="default"
        )
        for row in pq.read_table(path).to_pylist()
    }
