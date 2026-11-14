import pyarrow.parquet as pq
from bench import BenchRequest, ChatMessage
from benchman.datasets.dataset import Dataset, download_dataset_file


async def load() -> Dataset:
    path = await download_dataset_file(
        "google-research-datasets/mbpp",
        "full/test-00000-of-00001.parquet",
        revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    )
    return {
        str(row["task_id"]): BenchRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=row["text"] + "\n\nYour code should satisfy:\n" + "\n".join(row["test_list"]),
                )
            ],
            reasoning_effort="default",
        )
        for row in pq.read_table(path).to_pylist()
    }
