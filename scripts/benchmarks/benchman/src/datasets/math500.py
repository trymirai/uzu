import pyarrow.json as pj
from bench import BenchRequest, ChatMessage
from benchman.datasets.dataset import Dataset, download_dataset_file


async def load() -> Dataset:
    path = await download_dataset_file(
        "HuggingFaceH4/MATH-500", "test.jsonl", revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    )
    return {
        row["unique_id"]: BenchRequest(
            messages=[ChatMessage(role="user", content=row["problem"])], reasoning_effort="default"
        )
        for row in pj.read_json(path).to_pylist()
    }
