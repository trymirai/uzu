"""Benchmark Lucebox's luce_server on CUDA: Linux only, inside WSL on Windows.

luce_server speculates with a DFlash draft whose proposals it verifies as a DDTree. The adapter builds it at a pinned
commit, serves one model on a local port with every prompt cache off, and streams each run's chat completion.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated, Any

import typer
from bench import BenchRequest, BenchResponse, BenchSampling
from common import LOCAL_HOST, InferenceEngine, LocalServer, free_port, gpu_memory_used, run_loop
from huggingface_hub import hf_hub_download

PROJECT_DIR = Path(__file__).resolve().parent
# luce_server's stderr summary of each decode: the target forwards of a speculative decode, and the tokens of plain
# autoregressive decoding, which takes one forward per token.
SPEC_FORWARDS = re.compile(r"\[spec-decode\] target_forwards=(\d+) ")
AR_TOKENS = re.compile(r"\[ar-decode\] tokens=(\d+) ")


class LuceboxEngine(InferenceEngine):
    supports_reasoning_effort = True

    def __init__(
        self, model: str, draft_model: str | None, max_ctx: int, ddtree_budget: int, kv_cache_type: str | None
    ):
        super().__init__()
        binary = subprocess.run(
            ["bash", str(PROJECT_DIR / "bootstrap.sh")], check=True, stdout=subprocess.PIPE, text=True
        ).stdout.strip()
        port = free_port()
        command = [binary, resolve(model), "--host", LOCAL_HOST, "--port", str(port)]
        command += ["--max-ctx", str(max_ctx), "--default-max-tokens", str(max_ctx)]
        # Every run starts from an empty cache, as in the other adapters.
        command += ["--prefix-cache-slots", "0", "--no-agent-turn-cache", "--no-kv-cache"]
        # A reasoning effort turns on its thinking budget, which reserves 4096 reply tokens by closing thinking early.
        command += ["--hard-limit-reply-budget", "0"]
        if kv_cache_type is not None:
            command += ["--cache-type-k", kv_cache_type, "--cache-type-v", kv_cache_type]
        self.speculative = draft_model is not None
        if draft_model is not None:
            command += ["--draft", resolve(draft_model), "--ddtree", "--ddtree-budget", str(ddtree_budget)]

        self.graphics_baseline = gpu_memory_used()
        self.server = LocalServer(
            "luce_server",
            port,
            command,
            stdout=sys.stderr,
            stderr=subprocess.PIPE,
            text=True,
            # Its repetition guard would end a reply that repeats one token 32 times.
            env={"LUCE_DEGENERATE_RUN_TOKENS": "0"} | dict(os.environ),
        )
        self.decodes: queue.Queue[tuple[str, int]] = queue.Queue()
        threading.Thread(target=self.read_log, daemon=True).start()
        self.server.wait_ready()

    def read_log(self) -> None:
        assert self.server.process.stderr is not None
        for line in self.server.process.stderr:
            sys.stderr.write(line)
            if match := SPEC_FORWARDS.search(line):
                self.decodes.put(("spec", int(match[1])))
            elif match := AR_TOKENS.search(line):
                self.decodes.put(("ar", int(match[1])))

    def close(self) -> None:
        self.server.close()

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        num_runs = request.num_runs if request.num_runs is not None else 1
        if num_runs < 1:
            raise ValueError("num_runs must be 1 or greater")
        body = request_body(request)
        return [self.run_single(body) for _ in range(num_runs)]

    def run_single(self, body: dict[str, Any]) -> BenchResponse:
        while not self.decodes.empty():
            self.decodes.get_nowait()
        started = time.perf_counter()
        response = self.server.request("POST", "/v1/chat/completions", body)
        if response.status != 200:
            raise RuntimeError(f"luce_server answered {response.status}: {response.read().decode()}")
        pieces, usage = [], None
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line.removeprefix(b"data: ").strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"luce_server failed: {chunk['error']}")
            usage = chunk.get("usage") or usage
            for choice in chunk["choices"]:
                delta = choice["delta"]
                pieces += [delta.get("reasoning_content") or "", delta.get("content") or ""]
        duration = time.perf_counter() - started
        response.close()
        if usage is None:
            raise RuntimeError("luce_server sent no usage")

        timings = usage["timings"]
        if timings["cache_hit"] or usage["prompt_tokens_details"]["cached_tokens"]:
            raise RuntimeError("luce_server reused a cached prompt prefix")
        tokens = usage["completion_tokens"]
        # Server clocks: the prefill, then the decode from the prefill's first token to the last token.
        prefill = timings["prefill_ms"] / 1000
        decode = timings["decode_ms"] / 1000
        phys_footprint, resident = self.server.memory()
        return BenchResponse(
            text="".join(pieces),
            tokens_count=tokens,
            time_to_first_token=prefill,
            prompt_tps=usage["prompt_tokens"] / prefill,
            decode_tps=(tokens - 1) / decode if decode > 0 else 0.0,
            tokens_per_forward_pass=tokens / self.forward_passes(),
            duration=duration,
            memory_phys_footprint=phys_footprint,
            memory_resident=resident,
            memory_graphics_total=max(0, gpu_memory_used() - self.graphics_baseline),
        )

    def forward_passes(self) -> int:
        """The run's target forwards, which luce_server logs before its last chunk. A speculative decode counts the
        prefill, its verifications and the autoregressive tokens it may end with, logged before the verifications.
        Plain decoding counts its tokens: the prefill produces the first."""
        autoregressive = 0
        while True:
            try:
                kind, count = self.decodes.get(timeout=10)
            except queue.Empty:
                raise RuntimeError("luce_server logged no decode summary for the run") from None
            if kind == "spec":
                return 1 + count + autoregressive
            if not self.speculative:
                return count
            autoregressive += count


def request_body(request: BenchRequest) -> dict[str, Any]:
    if request.prompt_chat is None:
        raise ValueError("luce_server serves chat completions only; send prompt_chat")
    if request.speculative_depth is not None:
        raise ValueError("luce_server fixes speculation at load; omit speculative_depth")
    sampling = request.sampling or BenchSampling()
    if sampling.min_p:
        raise ValueError("luce_server has no min_p sampling")
    sampled = sampling.temp is not None and sampling.temp > 0
    body: dict[str, Any] = {
        "messages": [message.model_dump(mode="json", exclude_none=True) for message in request.prompt_chat],
        "stream": True,
        # luce_server renders Qwen chats without thinking unless a request asks for it; the other adapters keep the
        # template's default, which thinks.
        "chat_template_kwargs": {"enable_thinking": True}
        | ({"reasoning_effort": request.reasoning_effort} if request.reasoning_effort is not None else {}),
        # Explicit values: luce_server fills omitted ones from the model card, which samples at temperature 1.
        "temperature": sampling.temp if sampled else 0.0,
        "top_p": sampling.top_p if sampled and sampling.top_p else 1.0,
        "top_k": sampling.top_k if sampled and sampling.top_k else 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    body |= request.model_dump(mode="json", include={"tools", "tool_choice"})
    if request.max_tokens:
        body["max_tokens"] = request.max_tokens
    return body


def resolve(model: str) -> str:
    """A local GGUF file, or REPOSITORY:FILE.gguf downloaded from Hugging Face."""
    if (path := Path(model).expanduser()).is_file():
        return str(path)
    repository, _, filename = model.rpartition(":")
    if not repository or not filename.endswith(".gguf"):
        raise ValueError(f"{model}: expected a GGUF file or REPOSITORY:FILE.gguf")
    return hf_hub_download(repository, filename)


def run(
    model: Annotated[str, typer.Option("-m", "--model", help="Target GGUF: a path or REPOSITORY:FILE.gguf.")],
    draft_model: Annotated[
        str | None, typer.Option("-d", "--draft-model", help="DFlash draft GGUF: a path or REPOSITORY:FILE.gguf.")
    ] = None,
    max_ctx: Annotated[int, typer.Option(help="Context length luce_server allocates.")] = 16384,
    ddtree_budget: Annotated[int, typer.Option(help="DDTree verification candidates per step.")] = 22,
    kv_cache_type: Annotated[
        str | None,
        typer.Option(help="K and V cache element type, such as q8_0; luce_server's default for Qwen is q4_0."),
    ] = None,
) -> None:
    with redirect_stdout(sys.stderr):
        engine = LuceboxEngine(model, draft_model, max_ctx, ddtree_budget, kv_cache_type)
    try:
        run_loop(engine)
    finally:
        engine.close()


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
