import base64
import json
from hashlib import sha256
from typing import Any, Literal

import pyarrow.parquet as pq
from bench import BenchRequest, ChatMessage
from benchman.datasets.dataset import Dataset, download_dataset_file
from cryptography.fernet import Fernet

REVISIONS = {"fast": "9774e44ed35c339472520f8f76496419a8f90f9b", "slow": "5728d6abd8dde516df20a07b682543996e68d102"}
FERNET = Fernet(base64.urlsafe_b64encode(sha256(b"mirai labs").digest()))
MESSAGE_FIELDS = {"role", "content", "reasoning", "tool_calls", "tool_call_id", "name"}


def decrypt(value: str) -> Any:
    return json.loads(FERNET.decrypt(value))


def message(raw: dict[str, Any]) -> ChatMessage:
    if unknown := raw.keys() - MESSAGE_FIELDS:
        raise ValueError(f"Unknown message fields: {sorted(unknown)}")
    return ChatMessage(
        role=raw["role"],
        content=raw["content"],
        reasoning_content=raw.get("reasoning"),
        tool_calls=raw.get("tool_calls"),
        tool_call_id=raw.get("tool_call_id"),
    )


async def load(config: Literal["fast", "slow"]) -> Dataset:
    path = await download_dataset_file("trymirai/benchmarks-mixture", f"{config}.parquet", revision=REVISIONS[config])
    dataset = {}
    for row in pq.read_table(path).to_pylist():
        *context, reference = decrypt(row["messages"])
        if reference["role"] != "assistant":
            raise ValueError(f"{row['source']}:{row['source_id']} does not end with an assistant turn")
        dataset[f"{row['source']}:{row['source_id']}"] = BenchRequest(
            messages=[message(raw) for raw in context],
            tools=decrypt(row["tools"]),
            max_output_tokens=row["generation_cap"],
            reasoning_effort=row["reasoning_effort"],
        )
    return dataset
