import json
import urllib.request
from pathlib import Path

from data_exporter.model import Model

__all__ = [
    "BUNDLED_CONFIGS_PATH",
    "CONFIGS_PATH",
    "RESPONSES_PATH",
    "TOKENIZERS_PATH",
    "download_file",
    "load_registry",
]

BASE_PATH = Path(__file__).parent
ROOT_PATH = BASE_PATH.parent.parent

CRATES_PATH = ROOT_PATH / "crates"
HANASHI_PATH = CRATES_PATH / "hanashi"
BUNDLED_CONFIGS_PATH = HANASHI_PATH / "configs"

WORKSPACE_PATH = ROOT_PATH / "workspace"
DATA_PATH = WORKSPACE_PATH / "data"
CONFIGS_PATH = DATA_PATH / "configs"
TOKENIZERS_PATH = DATA_PATH / "tokenizers"
RESPONSES_PATH = DATA_PATH / "responses"


def load_registry() -> list[Model]:
    with open(DATA_PATH / "registry.json") as file:
        models = [Model.from_dict(entry) for entry in json.load(file)]
        return [model for model in models if len(model.encodings) > 0]


def download_file(url: str, output_path: Path) -> None:
    if not output_path.exists():
        urllib.request.urlretrieve(url, output_path)
