import sys
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from pathlib import Path

import huggingface_hub
from bench import BenchRequest, BenchResponse
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import TypeAdapter


class InferenceEngine(ABC):
    @abstractmethod
    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        pass


def get_model_path(model: str | Path) -> str:
    if isinstance(model, Path):
        return model.expanduser().as_posix()

    cache_dir = Path("~/.cache/huggingface/hub").expanduser()
    try:
        # skip huggingface info output if model already downloaded
        return huggingface_hub.snapshot_download(
            repo_id=model,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return huggingface_hub.snapshot_download(
            repo_id=model,
            cache_dir=cache_dir,
            local_files_only=False,
        )


def run_loop(engine: InferenceEngine) -> None:
    response_adapter = TypeAdapter(list[BenchResponse])

    for line in sys.stdin:
        if not line.strip():
            continue

        response_json: str
        try:
            request = BenchRequest.model_validate_json(line)
            with redirect_stdout(sys.stderr):
                responses: list[BenchResponse] = engine.execute(request)
            response_json = response_adapter.dump_json(responses).decode("utf-8")
        except Exception as error:  # noqa: BLE001
            print(f"Failed to process request: {error}", file=sys.stderr, flush=True)
            continue
        print(response_json, flush=True)
