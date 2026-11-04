import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx2
from bench import BenchResponse, ModelConfig, ReasoningEffort
from common import get_finish_reason
from huggingface_hub import HfApi, hf_hub_download
from mach import get_gpu_energy, get_memory_counters
from openai import OpenAI

from ..util import get_openai_base_url
from . import ServerEngine, get_openai_reasoning

SPLIT_GGUF = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def resolve_model(model: str, revision: str | None) -> Path:
    """Resolve a local GGUF or repo[:filename.gguf|quantization], including shards."""
    path = Path(model).expanduser()
    if path.exists() and revision is not None:
        raise ValueError(f"A local model has no revision: {model}")
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError("Magnitude requires a GGUF model file.")
        return path.absolute()
    if path.is_dir():
        files = sorted(str(file) for file in path.glob("*.gguf") if not file.name.lower().startswith("mmproj"))
        local = True
        repository, selector = "", ""
    else:
        if model.startswith(("/", "~", "./", "../")) or ":" not in model and model.lower().endswith(".gguf"):
            raise FileNotFoundError(f"GGUF model does not exist: {model}")
        repository, _, selector = model.partition(":")
        files = sorted(
            file
            for file in HfApi().list_repo_files(repository, revision=revision)
            if file.lower().endswith(".gguf") and not Path(file).name.lower().startswith("mmproj")
        )
        local = False

    # Keep split GGUFs together and launch their first shard. Preserve snapshot
    # symlinks so the engine can locate the other shards beside the first one.
    groups: dict[str, list[str]] = {}
    for file in files:
        split = SPLIT_GGUF.fullmatch(file)
        key = f"{split[1]}.gguf" if split else file
        groups.setdefault(key, []).append(file)
    matches = [parts for name, parts in groups.items() if not selector or selector == name or selector in parts]
    if selector and not matches and not selector.lower().endswith(".gguf"):
        quantization = re.compile(rf"(?:^|[-_.]){re.escape(selector)}$", re.IGNORECASE)
        matches = [parts for name, parts in groups.items() if quantization.search(Path(name).stem)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one GGUF model for {model!r}, found {len(matches)}. "
            "Use a local GGUF file or a Hugging Face repo:filename.gguf (or repo:quantization)."
        )
    parts = matches[0]
    split = SPLIT_GGUF.fullmatch(parts[0])
    if split:
        expected = [
            f"{split[1]}-{index:05d}-of-{split[3]}{Path(parts[0]).suffix}" for index in range(1, int(split[3]) + 1)
        ]
        if parts != expected:
            raise ValueError(f"Incomplete split GGUF model: {parts[0]}")
    if local:
        return Path(parts[0]).absolute()
    downloads = [hf_hub_download(repo_id=repository, filename=part, revision=revision) for part in parts]
    return Path(downloads[0]).absolute()


class MagnitudeConfig(ModelConfig):
    revision: str | None = None


class MagnitudeServerEngine(ServerEngine):
    """
    https://github.com/magnitudedev/magnitude
    """

    PROJECT_DIR = Path(__file__).resolve().parents[3] / "engine-magnitude"

    def __init__(self, host: str, port: int, config: MagnitudeConfig):
        super().__init__(host, port, config)
        self.revision = config.revision
        self.specdec = config.specdec

    def create_process(self) -> subprocess.Popen:
        model = resolve_model(self.model, self.revision)
        binary = os.environ.get("BENCHMARK_MAGNITUDE_BINARY")
        if binary is None:
            subprocess.run(
                ["bash", str(self.PROJECT_DIR / "bootstrap.sh")],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
            )
            binary = str(self.PROJECT_DIR / "target/release/magnitude-engine")
        return subprocess.Popen(
            [
                binary,
                "--model",
                str(model),
                "--no-projector",
                # Without a separate draft, the model's own MTP heads are the only way to speculate.
                "--method",
                "mtp" if self.specdec else "plain",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--served-model",
                self.model,
            ],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            start_new_session=True,
        )

    def get_reasoning(self, effort: ReasoningEffort, messages: list[Any], body: dict[str, Any]) -> dict[str, Any]:
        base = get_openai_base_url(self.host, self.port).copy_with(path="")
        response = httpx2.post(f"{base}/api/v1/models/{quote(self.model, safe='')}/properties", timeout=60)
        response.raise_for_status()
        reasoning = get_openai_reasoning(effort)
        if reasoning["reasoning_effort"] not in response.json()["reasoning"]["reasoning_efforts"]:
            raise ValueError(f"The model does not support reasoning_effort {effort}")
        return reasoning

    def handle_request(self, messages: list[Any], body: dict[str, Any]) -> BenchResponse:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Magnitude server is not running.")

        body.update(cache_prompt=False, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)
        with OpenAI(base_url=get_openai_base_url(self.host, self.port), api_key="not-needed") as client:
            energy_start = get_gpu_energy()
            started = time.perf_counter()
            response = client.chat.completions.create(model=self.model, messages=messages, extra_body=body)
            duration = time.perf_counter() - started
            energy_gpu = get_gpu_energy() - energy_start
        memory = get_memory_counters(pid=process.pid)
        # Magnitude infers in a worker process and reports what it charges each store, an SSD volume among them.
        observation = httpx2.get(f"{get_openai_base_url(self.host, self.port)}/memory", timeout=60)
        observation.raise_for_status()
        memory_used = sum(
            store[kind]
            for store in observation.json()["census"]["stores"]
            if "Volume" not in store["store"]
            for kind in ("model_bytes", "context_bytes", "compute_bytes", "auxiliary_bytes")
        )

        timings = (response.model_extra or {}).get("timings")
        if response.usage is None or not isinstance(timings, dict):
            raise RuntimeError("Magnitude response is missing usage or timing statistics.")

        return BenchResponse(
            text=response.choices[0].message.content or "",
            finish_reason=get_finish_reason(response.choices[0].finish_reason),
            input_tokens_count=response.usage.prompt_tokens,
            output_tokens_count=response.usage.completion_tokens,
            time_to_first_token=timings["time_to_first_token_ms"] / 1000,
            prefill_tps=timings["prompt_per_second"],
            decode_tps=timings["predicted_per_second"],
            # The API reports draft acceptance but no target-forward count.
            tokens_per_forward_pass=None,
            duration=duration,
            energy_gpu=energy_gpu,
            memory_used=memory_used,
            memory_phys_footprint=memory.phys_footprint,
            memory_resident=memory.resident_size,
            memory_graphics_total=memory.graphics_total,
        )
