import json
import time
from pathlib import Path
from typing import Any

import httpx2
import typer
from mach import AppleTempSensors, get_temp_sensors

MAX_TEMP = 60.0


def await_cooldown():
    temp: AppleTempSensors = get_temp_sensors()
    while temp.cpu_avg > MAX_TEMP or temp.gpu_avg > MAX_TEMP:
        time.sleep(1.0)
        temp = get_temp_sensors()


def get_openai_base_url(host: str, port: int) -> httpx2.URL:
    host = host.strip().removeprefix("[").removesuffix("]")
    if not host or any(character in host for character in "/?#@"):
        raise typer.BadParameter("Use a hostname or IP address without a URL scheme or path.", param_hint="--host")

    try:
        return httpx2.URL(scheme="http", host=host, port=port, path="/v1")
    except httpx2.InvalidURL as error:
        raise typer.BadParameter(str(error), param_hint="--host") from error


def load_input(path: Path) -> dict[str, Any]:
    text = path.expanduser().read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Input must be a JSON object.")
    if "messages" not in payload:
        raise ValueError("Input must contain 'messages'.")
    return payload
