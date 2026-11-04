import pyarrow.parquet as pq
from bench import BenchRequest, ChatMessage
from benchman.datasets.dataset import Dataset, download_dataset_file


async def load() -> Dataset:
    path = await download_dataset_file(
        "openai/gsm8k", "main/test-00000-of-00001.parquet", revision="740312add88f781978c0658806c59bc2815b9866"
    )
    return {
        str(index): BenchRequest(
            messages=[ChatMessage(role="user", content=row["question"])], reasoning_effort="default"
        )
        for index, row in enumerate(pq.read_table(path).to_pylist())
    }
