# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cryptography>=46",
#     "datasets>=3.4,<4",
#     "google-crc32c>=1.8",
#     "huggingface-hub>=1.6",
#     "jinja2>=3.1",
#     "psutil>=7.2",
#     "requests>=2.32",
#     "transformers==5.14.1",
#     "typer>=0.24",
# ]
# ///
"""Export performance-benchmarks' benchmarks-mixture-fast replay rows as benchmark requests.

performance-benchmarks' own loader decrypts, samples and orders the rows; the export stops unless their manifest digest
matches the one its Qwen3.8 comparison pins. Each line holds a row's chat, tools, generation cap and reasoning effort,
as performance-benchmarks replays them, and the prompt's length under the Qwen3.8 chat template, which sizes engine
contexts. The rows are decrypted gated data: keep them under the ignored workspace/ and never commit or share them. Run
it where Hugging Face is logged in with access to trymirai/benchmarks-mixture.
"""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

SUITE = "benchmarks-mixture-fast"
# comparisons/qwen38-uzu068-splash131.yaml in performance-benchmarks: 21 rows, seed 0, first turns only.
MANIFEST_DIGEST = "sha256:58c960cfdcd560e1b5f0afdf9394687320a0f6d15e6c09619e24e8c507a94763"


def main(
    performance_benchmarks: Annotated[Path, typer.Option(help="performance-benchmarks checkout.")],
    output: Annotated[Path, typer.Option()] = Path("workspace/mixture/benchmarks-mixture-fast.jsonl"),
    tokenizer: Annotated[str, typer.Option(help="Tokenizer whose chat template sizes contexts.")] = "Qwen/Qwen3.8-27B",
) -> None:
    sys.path.insert(0, str(performance_benchmarks.resolve()))
    from benchmarks.dataset_provenance import dataset_manifest
    from benchmarks.dataset_suite import load_dataset_suite, load_suite_completions
    from transformers import AutoTokenizer

    completions = load_suite_completions(load_dataset_suite(SUITE), limit=21, seed=0, first_turn_only=True)
    digest = dataset_manifest(SUITE, completions, first_turn_only=True)["manifest_digest"]
    if digest != MANIFEST_DIGEST:
        raise typer.BadParameter(f"{SUITE} rows have manifest digest {digest}, expected {MANIFEST_DIGEST}")

    template = AutoTokenizer.from_pretrained(tokenizer)
    lines = []
    for completion in completions:
        metadata = completion.source_metadata or {}
        messages = [message.to_chat_dict() for message in completion.context_messages or ()]
        prompt = template.apply_chat_template(
            templated(messages),
            tools=completion.tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            enable_thinking=True,
            reasoning_effort=metadata["reasoning_effort"],
        )
        request = {
            "prompt_chat": messages,
            "tools": completion.tools,
            "max_tokens": metadata["generation_cap"],
            "reasoning_effort": metadata["reasoning_effort"],
        }
        row = {"id": completion.id, "dataset": completion.dataset, "category": completion.category}
        lines.append(json.dumps(row | {"prompt_tokens": len(prompt), "request": request}) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines))
    typer.echo(f"{len(lines)} rows, {digest}, written to {output}", err=True)


def templated(messages: list[dict]) -> list[dict]:
    """Messages with JSON-string tool-call arguments parsed: Hugging Face's Qwen templates iterate them as objects."""
    result = []
    for message in messages:
        calls = [
            call | {"function": call["function"] | {"arguments": json.loads(call["function"]["arguments"])}}
            if isinstance(call.get("function", {}).get("arguments"), str)
            else call
            for call in message.get("tool_calls") or []
        ]
        result.append(message | {"tool_calls": calls} if calls else message)
    return result


if __name__ == "__main__":
    typer.run(main)
