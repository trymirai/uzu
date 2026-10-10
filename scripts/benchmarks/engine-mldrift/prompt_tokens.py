import json
import sys

from bench import BenchRequest
from common import get_tokenized_prompt
from transformers import AutoTokenizer


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    for line in sys.stdin:
        try:
            tokens = get_tokenized_prompt(BenchRequest.model_validate_json(line), tokenizer)
        except Exception as error:  # noqa: BLE001
            print(f"Failed to tokenize prompt: {error}", file=sys.stderr, flush=True)
            tokens = []
        print(json.dumps(tokens), flush=True)
