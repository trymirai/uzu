import asyncio
from pathlib import Path

from bench import BenchRequest
from huggingface_hub import hf_hub_download

Dataset = dict[str, BenchRequest]


async def download_dataset_file(repo_id: str, filename: str, *, revision: str) -> Path:
    return Path(await asyncio.to_thread(hf_hub_download, repo_id, filename, repo_type="dataset", revision=revision))
