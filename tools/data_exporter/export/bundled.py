import json
import re
from pathlib import Path

from rich.console import Console
from tokenizers import Tokenizer

from data_exporter.utils import BUNDLED_CONFIGS_PATH, CONFIGS_PATH, TOKENIZERS_PATH, load_registry

__all__ = ["export_bundled"]

BUNDLED_CONFIGS_MAPPING_PATH = BUNDLED_CONFIGS_PATH / "configs.json"


def load_bundled_configs_mapping() -> list[dict]:
    with open(BUNDLED_CONFIGS_MAPPING_PATH) as file:
        return json.load(file)


def resolve_hanashi_name(model_encodings: list[dict]) -> str | None:
    for encoding in model_encodings:
        if encoding.get("type") == "hanashi":
            return encoding.get("name")
    return None


def resolve_token_string(
    tokenizer_config: dict,
    model_config: dict,
    tokenizer: Tokenizer,
    token_key: str,
) -> str | None:
    token = tokenizer_config.get(f"{token_key}_token")
    if isinstance(token, dict):
        token = token["content"]
    if token is not None:
        return token
    token_id = model_config.get(f"{token_key}_token_id")
    if isinstance(token_id, list):
        token_id = token_id[-1] if token_id else None
    if token_id is not None:
        return tokenizer.decode([token_id])
    return None


def resolve_token_id(tokenizer: Tokenizer, token: str | None) -> int | None:
    if token is None:
        return None
    token_ids = tokenizer.encode(token, add_special_tokens=False).ids
    return token_ids[0] if token_ids else None


def resolve_stop_token_ids(
    model_config: dict,
    generation_config: dict | None,
    eos_token_id: int | None,
) -> list[int]:
    decoder_eos_token_ids = model_config.get("eos_token_id", [])
    if not isinstance(decoder_eos_token_ids, list):
        decoder_eos_token_ids = [decoder_eos_token_ids]

    generation_eos_token_ids = []
    if generation_config is not None:
        generation_eos_token_ids = generation_config.get("stop_token_ids", [])
        generation_eos_token_id = generation_config.get("eos_token_id")
        if isinstance(generation_eos_token_id, list):
            generation_eos_token_ids += generation_eos_token_id
        elif generation_eos_token_id is not None:
            generation_eos_token_ids.append(generation_eos_token_id)

    all_ids = decoder_eos_token_ids + generation_eos_token_ids + [eos_token_id]
    return sorted(set(all_ids))


def process_chat_template(template: str) -> str:
    generation_block_tag_regex = re.compile(r"{%-?\s*(?:generation|endgeneration)\s*-?%}")
    return generation_block_tag_regex.sub("", template)


def process_chat_template_model_specific(template: str, repo_id: str) -> str:
    if repo_id == "HuggingFaceTB/SmolLM3-3B":
        # Fix missing <|im_end|> for no-tools path: move it outside the tools if-block
        template = template.replace(
            '{{- "\\n\\n" -}}\n    {{- "<|im_end|>\\n" -}}\n  {%- endif -%}\n{%- endif -%}',
            '{{- "\\n\\n" -}}\n  {%- endif -%}\n  {{- "<|im_end|>\\n" -}}\n{%- endif -%}',
        )
    return template


def resolve_chat_template(model_path: Path, tokenizer_config: dict) -> str | None:
    template = tokenizer_config.get("chat_template")
    if template is not None:
        return template
    jinja_path = model_path / "chat_template.jinja"
    if jinja_path.exists():
        return jinja_path.read_text()
    chat_template_json = model_path / "chat_template.json"
    if chat_template_json.exists():
        with open(chat_template_json) as file:
            return json.load(file).get("chat_template")
    return None


def export_bundled(console: Console, err_console: Console) -> None:
    registry = load_registry()
    bundled_mapping = load_bundled_configs_mapping()

    seen_tokens: set[str] = set()
    seen_rendering: set[str] = set()

    for model in registry:
        hanashi_name = resolve_hanashi_name(model.encodings)
        if hanashi_name is None:
            continue

        mapping = next(
            (entry for entry in bundled_mapping if entry["name"] == hanashi_name),
            None,
        )
        if mapping is None:
            err_console.print(f"[red]{model.repo_id}: no bundled mapping for '{hanashi_name}'[/red]")
            continue

        tokens_name = mapping["tokens"]
        rendering_name = mapping["rendering"]
        model_path = CONFIGS_PATH / model.name
        model_tokenizer_path = CONFIGS_PATH / model.tokenizer_name

        if not model_path.exists():
            err_console.print(f"[red]{model.repo_id}: configs not exported yet[/red]")
            continue

        try:
            with open(model_path / "config.json") as file:
                model_config = json.load(file)

            with open(model_tokenizer_path / "tokenizer_config.json") as file:
                tokenizer_config = json.load(file)

            generation_config = None
            generation_config_path = model_tokenizer_path / "generation_config.json"
            if generation_config_path.exists():
                with open(generation_config_path) as file:
                    generation_config = json.load(file)

            tokenizer = Tokenizer.from_file(
                str(TOKENIZERS_PATH / model.tokenizer_name / "tokenizer.json"),
            )

            if tokens_name not in seen_tokens:
                seen_tokens.add(tokens_name)

                bos_token = resolve_token_string(tokenizer_config, model_config, tokenizer, "bos")
                eos_token = resolve_token_string(tokenizer_config, model_config, tokenizer, "eos")
                bos_token_id = resolve_token_id(tokenizer, bos_token)
                eos_token_id = resolve_token_id(tokenizer, eos_token)
                stop_token_ids = resolve_stop_token_ids(
                    model_config,
                    generation_config,
                    eos_token_id,
                )
                tokens_data = {
                    "bos_token_id": bos_token_id,
                    "eos_token_id": eos_token_id,
                    "stop_token_ids": stop_token_ids,
                    "banned_token_ids": [],
                }

                tokens_path = BUNDLED_CONFIGS_PATH / "tokens" / f"{tokens_name}.json"
                with open(tokens_path, "w") as file:
                    json.dump(tokens_data, file, indent=4)
                console.print(f"\t[dim]tokens/{tokens_name}.json updated[/dim]")

            if rendering_name not in seen_rendering:
                seen_rendering.add(rendering_name)

                chat_template = resolve_chat_template(model_path, tokenizer_config)
                if chat_template is None:
                    continue
                chat_template = process_chat_template(chat_template)
                chat_template = process_chat_template_model_specific(chat_template, model.repo_id)

                rendering_path = BUNDLED_CONFIGS_PATH / "rendering" / f"{rendering_name}.json"
                with open(rendering_path) as file:
                    rendering_config = json.load(file)
                rendering_config["jinja"]["template"] = chat_template
                with open(rendering_path, "w") as file:
                    json.dump(rendering_config, file, indent=4, ensure_ascii=False)
                console.print(f"\t[dim]rendering/{rendering_name}.json template updated[/dim]")

            console.print(f"[green]{model.repo_id} completed[/green]")
        except Exception as error:  # noqa: BLE001
            err_console.print(f"[red]{model.repo_id} failed[/red]: {error}")
