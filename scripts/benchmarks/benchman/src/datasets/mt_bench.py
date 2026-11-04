import pyarrow.parquet as pq
from bench import BenchRequest, ChatMessage
from benchman.datasets.dataset import Dataset, download_dataset_file


async def load() -> Dataset:
    path = await download_dataset_file(
        "HuggingFaceH4/mt_bench_prompts",
        "data/train-00000-of-00001-67c6c9fef07685a3.parquet",
        revision="e3a795c5e9a82ee40611c416b8a7786c73198991",
    )
    return {
        str(row["prompt_id"]): BenchRequest(
            messages=[ChatMessage(role="user", content=row["prompt"][0])], reasoning_effort="default"
        )
        for row in pq.read_table(path).to_pylist()
    }
