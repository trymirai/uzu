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

### [MLX](https://github.com/ml-explore/mlx-lm)
```bash
uv run --project engine-mlx bench-mlx -m "mlx-community/Qwen3.5-0.8B-4bit"
```

### [mlx-serve](https://github.com/ddalcu/mlx-serve)

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

The default suite also checks engine pins against the latest stable releases on PyPI and GitHub, including Python lockfiles.
