import huggingface_hub
from rich.console import Console

from data_exporter.utils import CONFIGS_PATH, load_registry

__all__ = ["export_configs"]

EXCLUDED_EXTENSIONS = {
    ".safetensors",
    ".pth",
    ".bin",
    ".gguf",
    ".ot",
    ".pt",
    ".h5",
    ".msgpack",
    ".onnx",
    ".tflite",
    ".litertlm",
    ".onnx_data",
}


def export_configs(console: Console, err_console: Console) -> None:
    CONFIGS_PATH.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    for model in registry:
        try:
            model_path = CONFIGS_PATH / model.name
            model_path.mkdir(parents=True, exist_ok=True)

            filenames = huggingface_hub.list_repo_files(model.repo_id)
            filenames = [
                filename
                for filename in filenames
                if not any(filename.endswith(extension) for extension in EXCLUDED_EXTENSIONS)
            ]

            for filename in filenames:
                file_path = model_path / filename
                if file_path.exists():
                    continue
                file_path.parent.mkdir(parents=True, exist_ok=True)
                huggingface_hub.hf_hub_download(
                    repo_id=model.repo_id,
                    filename=filename,
                    local_dir=model_path,
                )

            console.print(f"[green]{model.repo_id} completed ({len(filenames)} files)[/green]")
        except Exception as error:  # noqa: BLE001
            err_console.print(f"[red]{model.repo_id} failed[/red]: {error}")
