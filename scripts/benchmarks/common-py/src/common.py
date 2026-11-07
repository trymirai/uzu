import json
import socket
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from typing import Any, ClassVar

import huggingface_hub
from bench import BenchRequest, BenchResponse
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import TypeAdapter

REASONING_EFFORTS = ("low", "medium", "xhigh")
LOCAL_HOST = "127.0.0.1"


class InferenceEngine(ABC):
    # Engines that render a request's reasoning_effort opt in; run_loop rejects it for the rest.
    supports_reasoning_effort: ClassVar[bool] = False

    @abstractmethod
    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        pass


def get_model_path(model: str | Path) -> str:
    local = Path(model).expanduser()
    if isinstance(model, Path) or local.is_dir():
        return local.as_posix()

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


def get_tokenized_prompt(request: BenchRequest, tokenizer: Any) -> list[int]:
    prompt: list[int]
    if isinstance(request.prompt, str):
        prompt = tokenizer.encode(request.prompt)
    else:
        messages = [message.model_dump(mode="json") for message in request.prompt]
        options = request.model_dump(mode="json", include={"tools", "tool_choice"})
        if request.reasoning_effort is not None:
            options |= effort_template_options(tokenizer, request.reasoning_effort, messages, options)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True, **options
        )
    return prompt


def effort_template_options(tokenizer: Any, effort: str, messages: list[dict], options: dict) -> dict[str, Any]:
    """Chat template arguments for a reasoning effort. Like performance-benchmarks, reject a template that ignores the
    effort or renders it like another level."""
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {effort}")
    renders = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort=level,
            **options,
        )
        for level in REASONING_EFFORTS
    ]
    if renders.count(renders[REASONING_EFFORTS.index(effort)]) > 1:
        raise ValueError(f"The chat template renders reasoning effort {effort} like another level")
    return {"enable_thinking": True, "reasoning_effort": effort}


def process_memory(pid: int | str = "self") -> tuple[int, int]:
    """A Linux process's private anonymous memory including swap, as the footprint, and its resident set, from /proc."""
    fields = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("RssAnon", "VmSwap", "VmRSS"):
            fields[name] = int(value.split()[0]) * 1024
    return fields["RssAnon"] + fields.get("VmSwap", 0), fields["VmRSS"]


def run_loop(engine: InferenceEngine) -> None:
    response_adapter = TypeAdapter(list[BenchResponse])

    for line in sys.stdin:
        if not line.strip():
            continue

        response_json: str
        try:
            request = BenchRequest.model_validate_json(line)
            if request.reasoning_effort is not None and not engine.supports_reasoning_effort:
                raise ValueError("This engine does not support reasoning_effort")
            with redirect_stdout(sys.stderr):
                responses: list[BenchResponse] = engine.execute(request)
            response_json = response_adapter.dump_json(responses).decode("utf-8")
        except Exception as error:  # noqa: BLE001
            print(f"Failed to process request: {error}", file=sys.stderr, flush=True)
            continue
        print(response_json, flush=True)


class LocalServer:
    """A Linux inference server on `port`, which answers GET /health once its models are loaded. It gets SIGTERM if the
    adapter dies, so a killed benchmark never leaves it holding the GPU."""

    def __init__(self, name: str, port: int, command: list[str], **options: Any):
        self.name = name
        self.port = port
        self.process = subprocess.Popen(
            ["setpriv", "--pdeathsig", "TERM", *command], stdin=subprocess.DEVNULL, **options
        )

    def wait_ready(self) -> None:
        while True:
            if (code := self.process.poll()) is not None:
                raise RuntimeError(f"{self.name} exited with code {code} while loading")
            try:
                if self.request("GET", "/health", timeout=5).status == 200:
                    return
            except OSError:
                pass
            time.sleep(1)

    def request(self, method: str, path: str, body: Any = None, timeout: float | None = None) -> HTTPResponse:
        connection = HTTPConnection(LOCAL_HOST, self.port, timeout=timeout)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
        return connection.getresponse()

    def memory(self) -> tuple[int, int]:
        """`process_memory` summed over the server and its child processes."""
        pids, footprint, resident = [self.process.pid], 0, 0
        while pids:
            pid = pids.pop()
            for children in Path(f"/proc/{pid}/task").glob("*/children"):
                pids += [int(child) for child in children.read_text().split()]
            process_footprint, process_resident = process_memory(pid)
            footprint, resident = footprint + process_footprint, resident + process_resident
        return footprint, resident

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((LOCAL_HOST, 0))
        return probe.getsockname()[1]


def gpu_memory_used() -> int:
    """The GPU's used memory across all processes: WSL has no per-process GPU accounting."""
    output = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return int(output) * 2**20
