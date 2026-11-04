import asyncio
from pathlib import Path

from bench import BenchRequest
from huggingface_hub import hf_hub_download
from prefect.blocks.system import Secret

Dataset = dict[str, BenchRequest]


async def download_dataset_file(repo_id: str, filename: str, *, revision: str) -> Path:
    token = (await Secret.aload("hf-token")).get()
    path = await asyncio.to_thread(
        hf_hub_download, repo_id, filename, repo_type="dataset", revision=revision, token=token
    )
    return Path(path)
