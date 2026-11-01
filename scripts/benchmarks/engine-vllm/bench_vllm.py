import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated, Any

import typer
from bench import BenchRequest, BenchResponse
from common import InferenceEngine, get_model_path, get_tokenized_prompt, run_loop

# vLLM reads these when imported: logs stay off the JSON channel, and the engine core runs in this process so its
# timings and memory are this process's.
os.environ.setdefault("VLLM_LOGGING_STREAM", "ext://sys.stderr")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

DRAFTS = "vllm:spec_decode_num_drafts"


class VLLMEngine(InferenceEngine):
    def __init__(
        self,
        model: str,
        draft_model: str | None,
        speculative_depth: int | None,
        max_model_len: int,
        gpu_memory_utilization: float,
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
            # the KV cache. 8192 is vLLM's own default.
            max_num_batched_tokens=min(max_model_len, 8192),
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

        num_runs = request.num_runs if request.num_runs is not None else 1
        if num_runs < 1:
            raise ValueError("num_runs must be 1 or greater")
        if request.speculative_depth not in (None, self.speculative_depth):
            raise ValueError(f"vLLM fixes speculative_depth at load; this engine runs {self.speculative_depth}")

        prompt = get_tokenized_prompt(request, self.tokenizer)
        sampling = request.sampling
        temperature = sampling.temp if sampling is not None and sampling.temp is not None else 0.0
        params = SamplingParams(
            max_tokens=request.max_tokens or self.max_model_len - len(prompt),
            temperature=temperature,
            top_p=sampling.top_p if sampling is not None and sampling.top_p else 1.0,
            top_k=sampling.top_k if sampling is not None and sampling.top_k else 0,
            min_p=sampling.min_p if sampling is not None and sampling.min_p else 0.0,
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


def process_memory() -> tuple[int, int]:
    """Private anonymous memory including swap, as the footprint, and the resident set, from /proc."""
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("RssAnon", "VmSwap", "VmRSS"):
            fields[name] = int(value.split()[0]) * 1024
    return fields["RssAnon"] + fields.get("VmSwap", 0), fields["VmRSS"]


def graphics_memory() -> int:
    """The CUDA memory PyTorch reserved for this process at its peak, without the CUDA context: WSL exposes no
    per-process GPU accounting."""
    import torch

    return torch.cuda.max_memory_reserved()


def run(
    model: Annotated[str, typer.Option("-m", "--model")],
    draft_model: Annotated[
        str | None, typer.Option("-d", "--draft-model", help="DFlash draft checkpoint path or Hugging Face repository.")
    ] = None,
    speculative_depth: Annotated[
        int | None, typer.Option(help="Proposed tokens per step; defaults to the draft's DFlash block size minus one.")
    ] = None,
    max_model_len: Annotated[int, typer.Option(help="Context length the KV cache is sized for.")] = 16384,
    gpu_memory_utilization: Annotated[float, typer.Option(help="Share of GPU memory vLLM may reserve.")] = 0.9,
) -> None:
    with redirect_stdout(sys.stderr):
        engine = VLLMEngine(model, draft_model, speculative_depth, max_model_len, gpu_memory_utilization)
    run_loop(engine)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
