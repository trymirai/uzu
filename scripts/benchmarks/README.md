# Inference benchmarks

Benchmark adapters for inference engines. Each runner loads one model, reads requests from standard input, and writes generated text, timing, throughput, and process memory measurements as JSON Lines. A process can handle multiple requests without reloading its model.

Interaction with every engine follows the same loop: after launch, it waits for a request on stdin, runs inference, writes the result to stdout, and then waits for the next request.

Each engine includes adapter source files that collect the same set of benchmark metrics. Memory usage is measured using the shared API in [memory_counters.h](common-cpp/src/memory_counters.h).

Each engine supports:
* Argument `-m` or `--model` with value HuggingFace repository id or local model path.
* One shot execution: 
    ```bash
    ${RUN_COMMAND} -m "${MODEL}" <<'EOF' 
    ${INPUT_JSON} 
    EOF 
    ```
    for example
    ```bash
    ./engine-llamacpp/run.sh -m "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M" <<'EOF'
    {"prompt_text": "Tell me about London"}
    EOF
    ```

When `max_tokens` is omitted, `null`, or `0`, decoding continues until an EOS token.

`reasoning_effort` (`low`, `medium` or `xhigh`) sets a chat's reasoning effort as performance-benchmarks' replay
datasets do. uzu, llama.cpp, vLLM and Lucebox render it into the prompt and reject a chat template that ignores it; the
other engines reject the field.

## Engines

### [llama.cpp](https://github.com/ggml-org/llama.cpp)

```bash
./engine-llamacpp/run.sh -m "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M" 
```

Also DFlash is supported
```bash
./engine-llamacpp/run.sh -m "unsloth/Qwen3.6-27B-GGUF:Q4_K_S" \
    -d "ggml-org/Qwen3.6-27B-GGUF:BF16"
```

With a DFlash draft, a request that omits `speculative_depth` proposes the draft's block size minus one tokens per step;
`0` decodes without it. `--kv-cache-type` sets the target's and the draft's K and V caches by ggml type name (default
`f16`; `q8_0` for 8-bit). On Linux, `run.sh` builds with CUDA for the local GPU (CMake 3.31+); on Windows it runs inside
WSL, where `compare/` starts it through `wsl.exe`. There `memory_graphics_total` is the GPUs' used memory above what was
in use before the model loaded, because Linux has no per-process GPU accounting.

### [ML Drift](https://github.com/google-ai-edge/ml-drift)

```bash
./engine-mldrift/run.sh -m "Qwen/Qwen3-0.6B"
```

Requires `bazelisk` (`brew install bazelisk`) and, for Hugging Face ids, the `hf` CLI. The Metal LLM sample runner supports the `gemma3:270m`, `gemma3:1b`, `gemma4:12b`, `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:8b`, and `qwen3:14b` presets (the last added by the benchmark patch), selected from the checkpoint's `config.json`. The first run of a checkpoint converts its float weights with upstream `extract_weights_hf` through Bazel: to Q4_0 (block size 32) by default, or unquantized with `--precision f16`.

Prompts are tokenized by ML Drift's own tokenizer. Chats use its turn markers for user, assistant and system turns (Gemma 3 has no system turn); tools, tool calls and reasoning content are rejected. Decoding is greedy only, so requests with `sampling` or a positive `speculative_depth` are rejected. The prefill graph is compiled for each prompt length, so each request rebuilds the runner before its first run.

Compare against uzu on the same Qwen3 checkpoint at 4 bits (uzu runs a lalamo conversion of `mlx-community/<model>-4bit`, ML Drift its Q4_0 extraction), or at 16 bits with `--bits 16`:
```bash
uv run compare/uzu_mldrift.py --model Qwen/Qwen3-14B --bits 4 --num-runs 5 --max-tokens 256
```

### [MLX](https://github.com/ml-explore/mlx-lm)
```bash
uv run --project engine-mlx bench-mlx -m "mlx-community/Qwen3.5-0.8B-4bit"
```

### [mlx-serve](https://github.com/ddalcu/mlx-serve)

Requires Zig 0.17, pkg-config and libwebp.

```bash
./engine-mlxserve/run.sh -m "mlx-community/Qwen3.5-0.8B-4bit"
```

Supported native MTP heads are detected and loaded automatically from the target checkpoint, including sidecars. A detected head that fails to load or bind produces an error. Models without a supported head use ordinary autoregressive decoding.

Pass a compatible DFlash or Gemma assistant checkpoint with `--draft-model` (or `-d`) to use external speculative decoding. Both paths accept local directories or Hugging Face repositories; an explicit draft model takes precedence over MTP.

```bash
./engine-mlxserve/run.sh -m "mlx-community/Qwen3.6-27B-4bit" \
    --draft-model "z-lab/Qwen3.6-27B-DFlash"
```

Omit `speculative_depth` in the JSON request for the decoder's default depth, set it to `0` for ordinary decoding, or supply a positive depth within the decoder's supported range. DFlash uses upstream's hardware-dependent default block size. Depth counts proposed tokens; a DFlash verification block contains one additional target token. Upstream's adaptive acceptance gates can reduce speculation during a request. Inference runs directly through Zig sources, without HTTP.

`tokens_per_forward_pass` counts emitted tokens per target forward, including prefill, verification, and any replay. Draft-head forwards are excluded. Each run starts with fresh generation caches and adaptive timing/acceptance history.

### [MTPLX](https://github.com/youssofal/mtplx)
```bash
uv run --project engine-mtplx bench-mtplx -m "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed"
```

MTPLX uses the model's multi-token prediction (MTP) heads to draft tokens and verifies them with the target model: this is speculative decoding without a separate draft model. Omit `speculative_depth` to use the model pack's recommended depth (fallback: 3), set it to `0` for ordinary autoregressive decoding, or supply a positive depth to override it. Positive depths require matching MTP weights.

### [oMLX](https://github.com/jundot/omlx)
```bash
uv run --project engine-omlx bench-omlx -m "mlx-community/SmolLM-135M-Instruct-4bit"
```

Native MTP is enabled automatically when the target checkpoint has embedded prediction heads supported by oMLX's text engine.

```bash
uv run --project engine-omlx bench-omlx -m "TheWirelessPhoenix/Qwen3.5-4B-oQ4e-mtp"
```

Also DFlash is supported
```bash
uv run --project engine-omlx bench-omlx -m "mlx-community/Qwen3.6-27B-4bit" \
    -d "z-lab/Qwen3.6-27B-DFlash"
```

### [splash](https://github.com/incoai/splash)

```bash
uv run --project engine-splash bench-splash -m "unsloth/Qwen3.8-27B-GGUF:UD-IQ1_S"
```

DFlash already supported with hardcoded `speculative_depth == 7`

### [uzu](https://github.com/trymirai/uzu)

```bash
cargo run --release -p benchmarks-uzu -- -m "trymirai/LFM2.5-230M-M"
```

Current configuration enables DFlash only on M5, M5 Pro, and M5 Max
```bash
cargo run --release -p benchmarks-uzu -- -m "trymirai/Qwen3.5-9B-M"
```

On Windows the adapter runs on Vulkan. Its memory fields are the process's private commit (`memory_phys_footprint`),
working set (`memory_resident`) and the local video memory DXGI attributes to it on its GPU
(`memory_graphics_total`): the largest across hardware adapters, since DXGI can list one GPU under several LUIDs, as a
Remote Desktop session does. Linux builds only report that they are unsupported.

### [vLLM](https://github.com/vllm-project/vllm)

Linux and CUDA only; on Windows it runs inside WSL. Install [uv](https://docs.astral.sh/uv/) in WSL first.
```bash
uv run --project engine-vllm bench-vllm -m "Qwen/Qwen3-0.6B"
```

Pass a DFlash draft with `--draft-model` (or `-d`) for speculative decoding. The proposed tokens per step default to
the draft's DFlash block size minus one; `--speculative-depth` overrides them for the engine's lifetime, so requests may
only omit `speculative_depth` or repeat that value. `--max-model-len` (default 16384), `--gpu-memory-utilization`
(default 0.9) and the prefill chunk `--max-num-batched-tokens` (default 8192; smaller chunks leave more memory) size the
KV cache.
```bash
uv run --project engine-vllm bench-vllm -m "Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM" -d "syvai/Qwen3.8-27B-DFlash2-W4A16"
```

The engine core runs in the adapter's process with prefix caching off. Time to first token and decode throughput come
from vLLM's own per-request timestamps (arrival to first token, then first to last token). `tokens_per_forward_pass`
divides the generated tokens by the target forwards: the prefill plus one verification per draft, from vLLM's
speculative-decoding counters. `memory_graphics_total` is the peak CUDA memory PyTorch reserved, without the CUDA context,
because WSL has no per-process GPU accounting, so it shows the pool `--gpu-memory-utilization` reserves up front, not
the memory vLLM uses.

From Windows, the tests and `compare/` start it through `wsl.exe` with its virtual environment under
`~/.cache/uzu-benchmarks` inside WSL, so model paths must be Hugging Face ids or paths WSL can open.

On a 24 GB GPU, stock vLLM leaves a 4-bit Qwen3.8-27B and its DFlash2 draft room for about 8K tokens of context.
`--hyperqwen` serves [HyperQwen](https://github.com/syv-ai/HyperQwen) instead: vLLM 0.32.0 with syv-ai's patches, their
Qwen3.8-27B with 4-bit lm_head and embeddings, and the W4A16 DFlash2 drafter, in its single-user DFlash2 profile (7
proposed tokens per step, lookup drafting). It runs uzu's precisions: INT8 activations on the 4-bit weights (W4A8
Marlin, `INT8_ACT=int8`) and an INT8 per-token-head KV cache, which its long-context arm reads with the pth-flash-attn
kernel (`CTX=long PTH_FA=1`, 128K tokens of context). `engine-vllm/hyperqwen.sh` installs it at a pinned commit under
`~/.cache/uzu-benchmarks` the first time (about 40 GB). The adapter starts its launcher with prefix caching off, renders
prompts like the in-process engine and streams completions of their tokens. Time to first token and decode throughput are
client clocks on that stream, `tokens_per_forward_pass` comes from the server's speculative-decoding counters, and
`memory_graphics_total` is the GPU's used memory above what was in use before the server started.
```bash
uv run --project engine-vllm bench-vllm --hyperqwen
```

### [Lucebox](https://github.com/Luce-Org/lucebox)

Linux and CUDA only; on Windows it runs inside WSL like vLLM. The first run builds `luce_server` at a pinned commit for
the local GPU under `~/.cache/uzu-benchmarks/lucebox` (CUDA 12+, CMake 3.21+). Models are GGUF files, given as paths or
`REPOSITORY:FILE.gguf`:
```bash
uv run --project engine-lucebox bench-lucebox \
    -m "Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF:Qwen3.8-27B-IQ4_XS-pure.gguf" \
    -d "Lucebox/Qwen3.8-27B-DFlash2-GGUF:Qwen3.8-27B-DFlash2-Q8_0.gguf"
```

The adapter serves the model on a local port with every prompt cache off and streams chat completions; luce_server has
no raw-text route, so requests need `prompt_chat`. It verifies the DFlash draft's proposals as a tree (`--ddtree-budget`,
default 22: the fastest on an RTX 4090 for this Qwen3.8 pair, ahead of 16, 28 and 36). Chats render the template's
default, which thinks for Qwen, although luce_server alone leaves thinking off, and its reply budget, which would close
thinking early once a request sets a reasoning effort, is off. Requests without sampling decode greedily: the adapter
sends every sampling field, since luce_server fills omitted ones from the model card. Time to first token and decode throughput come from
luce_server's prefill and decode timers (first to last token). `tokens_per_forward_pass` divides the generated tokens by
the prefill plus the target forwards luce_server logs for the request. `memory_graphics_total` is the GPU's used memory
above what was in use before the server started, because WSL has no per-process GPU accounting. `--kv-cache-type` sets
its K and V caches (luce_server keeps Qwen's at `q4_0` by default; its model card serves this pair at `q8_0`).

## iOS

[platform-ios](platform-ios) runs these engines inside the [iOS app](../../apps/benchmarks-ios) on an iPhone or iPad:
```bash
uv run --project platform-ios bench-ios run -e llamacpp -m "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M"
```
Setup, device selection, model downloads and pushes are in the [app's README](../../apps/benchmarks-ios/README.md). `pytest engines` leaves it out because it needs a device; test it with `uv run pytest --engine ios --ios-engine ENGINE --ios-model MODEL`, adding `--ios-device UDID` or `--ios-host HOST` as needed.

## Compare

[./compare](./compare) runs the same requests through several engines and prints their averages. Each engine first
answers one run of the first request as a warmup, then, unless `--no-wait-cooldown`, waits for the CPU and GPU (macOS)
or the NVIDIA GPU (elsewhere) to cool to 60°C before each measured request.

uzu against llama.cpp, vLLM and Lucebox on one NVIDIA GPU, each with its own speculative decoding and a 4-bit
Qwen3.8-27B, on Windows only: uzu runs natively and the others in WSL, one after another; `--engine NAME` (repeatable)
picks some of them. vLLM runs as HyperQwen (`bench-vllm --hyperqwen`). As far as each engine allows, all run uzu's
precisions: 4-bit weights of about its 13.5 GiB (llama.cpp shares Lucebox's IQ4_XS GGUF), INT8 activations (ggml
quantizes them to 8 bits in its quantized matmuls; HyperQwen runs W4A8) and an 8-bit KV cache (`q8_0` in ggml,
HyperQwen's INT8 per token and head like uzu's). With task files (`--input`, default london-summary), engines decode
greedily up to `--max-tokens` unless `--temperature` is set:
```bash
uv run compare/uzu_nvidia.py
```

`--rows` replays performance-benchmarks' benchmarks-mixture-fast instead. Export its 21 rows once where Hugging Face is
logged in with access to the gated `trymirai/benchmarks-mixture` (WSL on this machine). The export checks them against
the manifest digest of performance-benchmarks' Qwen3.8 comparison and writes decrypted rows, which must stay in the
ignored `workspace/`:
```bash
uv run compare/mixture_export.py --performance-benchmarks PATH/TO/performance-benchmarks
```

Each row carries its chat, tools, reasoning effort and generation cap, as performance-benchmarks replays them. Engines
that size their context at load get the largest row's prompt, cap and 64 tokens, like performance-benchmarks'
`context_size: auto` for each row; llama.cpp sizes every run itself, and HyperQwen serves 128K tokens. With the
settings of performance-benchmarks' Qwen3.8 comparison:
```bash
uv run compare/uzu_nvidia.py --rows workspace/mixture/benchmarks-mixture-fast.jsonl \
    --num-runs 5 --temperature 1 --top-p 0.95 --top-k 20
```

The summary gives each engine's decode rate as the mean over runs, which weighs rows equally like performance-benchmarks'
dataset average, and token-weighted. `workspace/compare-results/` keeps every run's measurements without the generated
text.

## Server

[./server](./server) contains tool for easy benchmarking OpenAI-compatible server
```
uv run bench-server uzu \
    --model "trymirai/Qwen3.5-4B-M" \
    --input server/tasks/ai-history-summary.json \
    --num-runs 3
```
Output will show average metrics.
By default, each run waits until the CPU and GPU (macOS) or the NVIDIA GPU (elsewhere) are at or below 60°C.
Add `--no-wait-cooldown` to send requests without waiting for cooling.

TensorFold requires Zig 0.17.

## Tests
Unit tests
```bash
uv run pytest
```

All engines integration tests
```bash
uv run pytest engines
```

Specific engine test
```bash
uv run pytest --engine uzu
```

`pytest engines` leaves vLLM and Lucebox out because they need CUDA; test them with `uv run pytest --engine vllm` and
`--engine lucebox`. On Windows, `--engine uzu` builds and runs uzu natively, and the other two run inside WSL.
