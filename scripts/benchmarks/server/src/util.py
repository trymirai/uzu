import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx2
import typer

MAX_TEMP = 60.0


def read_temperatures() -> dict[str, float]:
    """Average CPU and GPU temperatures from Apple's sensors on macOS; elsewhere the hottest NVIDIA GPU, as nvidia-smi
    reports it, since Windows exposes CPU sensors only to administrators."""
    if sys.platform == "darwin":
        from mach import get_temp_sensors

        temp = get_temp_sensors()
        return {"CPU": temp.cpu_avg, "GPU": temp.gpu_avg}
    query = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    return {"GPU": max(float(value) for value in output.split())}


def await_cooldown():
    temperatures = read_temperatures()
    if max(temperatures.values()) > MAX_TEMP:
        readings = ", ".join(f"{name} {value:.1f}°C" for name, value in temperatures.items())
        typer.echo(f"Waiting for cooldown: {readings} (target <= {MAX_TEMP:.0f}°C).", err=True)
    while max(temperatures.values()) > MAX_TEMP:
        time.sleep(1.0)
        temperatures = read_temperatures()


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
