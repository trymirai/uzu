import json
from pathlib import Path

from rich.console import Console
from tokenizers import AddedToken, Tokenizer

from data_exporter.utils import CONFIGS_PATH, TOKENIZERS_PATH, download_file, load_registry

__all__ = ["export_tokenizers"]


def export_tokenizers(console: Console, err_console: Console) -> None:
    TOKENIZERS_PATH.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    for model in registry:
        try:
            model_tokenizer_path = TOKENIZERS_PATH / model.tokenizer_name
            model_tokenizer_path.mkdir(parents=True, exist_ok=True)

            tokenizer_path = model_tokenizer_path / "tokenizer.json"
            if not tokenizer_path.exists():
                model_path = CONFIGS_PATH / model.tokenizer_name

                with open(model_path / "tokenizer_config.json") as file:
                    tokenizer_config = json.load(file)

                tokenizer_json_path = model_path / "tokenizer.json"
                tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
                with open(tokenizer_json_path) as file:
                    tokenizer_json = json.load(file)

                added_tokens = tokenizer_config.get("added_tokens_decoder", {}).values()
                normalizer = tokenizer_json.get("normalizer")
                has_prepend_normalizer = isinstance(normalizer, dict) and any(
                    step.get("type") == "Prepend" for step in normalizer.get("normalizers", [])
                )

                if has_prepend_normalizer:
                    console.print(f"[yellow]{model.repo_id} has prepend normalizer[/yellow]")

                added_tokens = [
                    AddedToken(
                        content=token["content"],
                        special=token["special"],
                        normalized=False if has_prepend_normalizer else token["normalized"],
                        lstrip=token["lstrip"],
                        rstrip=token["rstrip"],
                        single_word=token["single_word"],
                    )
                    for token in added_tokens
                ]
                added_special_tokens = [token for token in added_tokens if token.special]
                added_not_special_tokens = [token for token in added_tokens if not token.special]
                tokenizer.add_special_tokens(added_special_tokens)
                tokenizer.add_tokens(added_not_special_tokens)
                tokenizer.save(str(tokenizer_path))

            if model.harmony is not None:
                filename = Path(model.harmony.tokenizer_url).name
                download_file(
                    model.harmony.tokenizer_url,
                    model_tokenizer_path / filename,
                )

            console.print(f"[green]{model.repo_id} completed[/green]")
        except Exception as error:  # noqa: BLE001
            err_console.print(f"[red]{model.repo_id} failed[/red]: {error}")
