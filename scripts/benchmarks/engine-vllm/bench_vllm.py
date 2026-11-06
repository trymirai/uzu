import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated, Any

import typer
from bench import BenchRequest, BenchResponse
from common import (
    InferenceEngine,
    LocalServer,
    free_port,
    get_model_path,
    get_tokenized_prompt,
    gpu_memory_used,
    process_memory,
    run_loop,
)

# vLLM reads these when imported: logs stay off the JSON channel, and the engine core runs in this process so its
# timings and memory are this process's.
os.environ.setdefault("VLLM_LOGGING_STREAM", "ext://sys.stderr")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

PROJECT_DIR = Path(__file__).resolve().parent
DRAFTS = "vllm:spec_decode_num_drafts"
# The checkpoint hyperqwen.sh requantizes. HyperQwen serves its variant with a 4-bit lm_head, with the same tokenizer.
HYPERQWEN_MODEL = "models/Qwen3.8-27B-W4A16-AutoRound"
HYPERQWEN_NAME = "qwen3.8-27b"


class VLLMEngine(InferenceEngine):
    supports_reasoning_effort = True

    def __init__(
        self,
        model: str,
        draft_model: str | None,
        speculative_depth: int | None,
        max_model_len: int,
        gpu_memory_utilization: float,
        max_num_batched_tokens: int,
    ):
        super().__init__()
        from vllm import LLM

        speculative_config = None
        self.speculative_depth = 0
        if draft_model is not None:
            draft_path = get_model_path(draft_model)
            if speculative_depth is None:
                # A DFlash verification block holds the proposed tokens and one target token.
                block_size = json.loads((Path(draft_path) / "config.json").read_text())["dflash_config"]["block_size"]
                speculative_depth = block_size - 1
            self.speculative_depth = speculative_depth
            speculative_config = {"method": "dflash", "model": draft_path, "num_speculative_tokens": speculative_depth}
        elif speculative_depth:
            raise ValueError("A positive speculative depth requires --draft-model")

        self.max_model_len = max_model_len
        self.llm = LLM(
            model=get_model_path(model),
            speculative_config=speculative_config,
            max_model_len=max_model_len,
            # One sequence never batches more than its context; a smaller budget shrinks the profiling run that sizes
            # the KV cache.
            max_num_batched_tokens=min(max_model_len, max_num_batched_tokens),
            max_num_seqs=1,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            language_model_only=True,
            disable_log_stats=False,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def drafts(self) -> int:
        from vllm.v1.metrics.reader import Counter

        metrics = self.llm.get_metrics()
        return sum(metric.value for metric in metrics if isinstance(metric, Counter) and metric.name == DRAFTS)

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        num_runs = get_num_runs(request)
        if request.speculative_depth not in (None, self.speculative_depth):
            raise ValueError(f"vLLM fixes speculative_depth at load; this engine runs {self.speculative_depth}")

        prompt = get_tokenized_prompt(request, self.tokenizer)
        params = SamplingParams(
            max_tokens=request.max_tokens or self.max_model_len - len(prompt), **sampling_options(request)
        )

        responses = []
        for _ in range(num_runs):
            drafts = self.drafts()
            started = time.perf_counter()
            output: Any = self.llm.generate(TokensPrompt(prompt_token_ids=prompt), params, use_tqdm=False)[0]
            duration = time.perf_counter() - started
            # Engine-core timestamps: arrival to first token, then first to last token.
            stats = output.metrics
            completion = output.outputs[0]
            tokens = len(completion.token_ids)
            # The prefill forward, then one target forward per verified draft, or per token without a drafter.
            forward_passes = 1 + (self.drafts() - drafts if self.speculative_depth else tokens - 1)
            decode = stats.last_token_ts - stats.first_token_ts
            phys_footprint, resident = process_memory()
            responses.append(
                BenchResponse(
                    text=completion.text,
                    tokens_count=tokens,
                    time_to_first_token=stats.first_token_latency,
                    prompt_tps=len(prompt) / stats.first_token_latency,
                    decode_tps=(tokens - 1) / decode if decode > 0 else 0.0,
                    tokens_per_forward_pass=tokens / forward_passes,
                    duration=duration,
                    memory_phys_footprint=phys_footprint,
                    memory_resident=resident,
                    memory_graphics_total=graphics_memory(),
                )
            )
        return responses


class HyperQwenEngine(InferenceEngine):
    """HyperQwen's single-user DFlash2 server: vLLM 0.32.0 with syv-ai's patches, serving their requantized Qwen3.8-27B
    with the W4A16 DFlash2 drafter. Prompts render as in VLLMEngine and go to the server as tokens, with prefix caching
    off as in the other adapters."""

    supports_reasoning_effort = True

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        directory = Path(
            subprocess.run(
                ["bash", str(PROJECT_DIR / "hyperqwen.sh")], check=True, stdout=subprocess.PIPE, text=True
            ).stdout.strip()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(directory / HYPERQWEN_MODEL)
        port = free_port()
        self.graphics_baseline = gpu_memory_used()
        self.server = LocalServer(
            "HyperQwen",
            port,
            ["bash", "single-user/start_qwen.sh"],
            cwd=directory,
            stdout=sys.stderr,
            env=os.environ | {"SPEC": "dflash2", "PREFIX_CACHE": "0", "PORT": str(port)},
        )
        self.server.wait_ready()

    def close(self) -> None:
        self.server.close()

    def drafts(self) -> float:
        metrics = self.server.request("GET", "/metrics").read().decode()
        return sum(
            float(line.rpartition(" ")[2]) for line in metrics.splitlines() if line.startswith(f"{DRAFTS}_total")
        )

    def execute(self, request: BenchRequest) -> list[BenchResponse]:
        num_runs = get_num_runs(request)
        if request.speculative_depth is not None:
            raise ValueError("HyperQwen fixes speculation at load; omit speculative_depth")
        prompt = get_tokenized_prompt(request, self.tokenizer)
        body = {
            "model": HYPERQWEN_NAME,
            "prompt": prompt,
            "max_tokens": request.max_tokens or None,
            "stream": True,
            "stream_options": {"include_usage": True},
        } | sampling_options(request)
        return [self.run_single(body, len(prompt)) for _ in range(num_runs)]

    def run_single(self, body: dict[str, Any], prompt_tokens: int) -> BenchResponse:
        drafts = self.drafts()
        started = time.perf_counter()
        response = self.server.request("POST", "/v1/completions", body)
        if response.status != 200:
            raise RuntimeError(f"HyperQwen answered {response.status}: {response.read().decode()}")
        pieces, usage, first_token, last_token = [], None, None, None
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line.removeprefix(b"data: ").strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            if chunk["choices"]:
                # Client clocks: the server streams each step's tokens as it decodes them.
                last_token = time.perf_counter()
                first_token = first_token or last_token
                pieces += [choice["text"] for choice in chunk["choices"]]
        duration = time.perf_counter() - started
        response.close()
        if usage is None or first_token is None or last_token is None:
            raise RuntimeError("HyperQwen streamed no tokens or no usage")
        if usage["prompt_tokens"] != prompt_tokens or (usage.get("prompt_tokens_details") or {}).get("cached_tokens"):
            raise RuntimeError("HyperQwen changed or reused the prompt")

        tokens = usage["completion_tokens"]
        time_to_first_token = first_token - started
        decode = last_token - first_token
        phys_footprint, resident = self.server.memory()
        return BenchResponse(
            text="".join(pieces),
            tokens_count=tokens,
            time_to_first_token=time_to_first_token,
            prompt_tps=prompt_tokens / time_to_first_token,
            decode_tps=(tokens - 1) / decode if decode > 0 else 0.0,
            # The prefill forward, then one target forward per verified draft.
            tokens_per_forward_pass=tokens / (1 + self.drafts() - drafts),
            duration=duration,
            memory_phys_footprint=phys_footprint,
            memory_resident=resident,
            memory_graphics_total=max(0, gpu_memory_used() - self.graphics_baseline),
        )


def get_num_runs(request: BenchRequest) -> int:
    num_runs = request.num_runs if request.num_runs is not None else 1
    if num_runs < 1:
        raise ValueError("num_runs must be 1 or greater")
    return num_runs


def sampling_options(request: BenchRequest) -> dict[str, float]:
    """vLLM sampling parameters: greedy without sampling, and no penalties, which the server would otherwise take from
    the model's generation config."""
    sampling = request.sampling
    return {
        "temperature": sampling.temp if sampling is not None and sampling.temp is not None else 0.0,
        "top_p": sampling.top_p if sampling is not None and sampling.top_p else 1.0,
        "top_k": sampling.top_k if sampling is not None and sampling.top_k else 0,
        "min_p": sampling.min_p if sampling is not None and sampling.min_p else 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }


def graphics_memory() -> int:
    """The CUDA memory PyTorch reserved for this process at its peak, without the CUDA context: WSL exposes no
    per-process GPU accounting."""
    import torch

    return torch.cuda.max_memory_reserved()


def run(
    model: Annotated[str | None, typer.Option("-m", "--model")] = None,
    draft_model: Annotated[
        str | None, typer.Option("-d", "--draft-model", help="DFlash draft checkpoint path or Hugging Face repository.")
    ] = None,
    speculative_depth: Annotated[
        int | None, typer.Option(help="Proposed tokens per step; defaults to the draft's DFlash block size minus one.")
    ] = None,
    max_model_len: Annotated[int, typer.Option(help="Context length the KV cache is sized for.")] = 16384,
    gpu_memory_utilization: Annotated[float, typer.Option(help="Share of GPU memory vLLM may reserve.")] = 0.9,
    max_num_batched_tokens: Annotated[
        int, typer.Option(help="Prefill chunk; smaller chunks leave the KV cache more memory. vLLM's default: 8192.")
    ] = 8192,
    hyperqwen: Annotated[
        bool,
        typer.Option(help="Serve HyperQwen: its patched vLLM, checkpoint and DFlash2 draft, instead of --model."),
    ] = False,
) -> None:
    if hyperqwen:
        if model is not None or draft_model is not None:
            raise typer.BadParameter("--hyperqwen serves its own model and draft")
        with redirect_stdout(sys.stderr):
            server = HyperQwenEngine()
        try:
            run_loop(server)
        finally:
            server.close()
        return
    if model is None:
        raise typer.BadParameter("Pass --model or --hyperqwen")
    with redirect_stdout(sys.stderr):
        engine = VLLMEngine(
            model, draft_model, speculative_depth, max_model_len, gpu_memory_utilization, max_num_batched_tokens
        )
    run_loop(engine)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
