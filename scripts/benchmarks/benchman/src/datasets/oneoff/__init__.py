from pathlib import Path

from bench import BenchRequest
from benchman.datasets.dataset import Dataset

DIRECTORY = Path(__file__).parent
ONEOFFS = {path.stem: BenchRequest.model_validate_json(path.read_text()) for path in sorted(DIRECTORY.glob("*.json"))}


async def load() -> Dataset:
    return ONEOFFS
