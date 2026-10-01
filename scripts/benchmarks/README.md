# Inference benchmarks

Benchmark adapters for inference engines. Each runner loads one model, reads requests from standard input, and writes generated text, timing, throughput, and process memory measurements as JSON Lines. A process can handle multiple requests without reloading its model.

Interaction with every engine follows the same loop: after launch, it waits for a request on stdin, runs inference, writes the result to stdout, and then waits for the next request.

Each engine includes adapter source files that collect the same set of benchmark metrics. Memory usage is measured using the shared API in [memory_counters.h](common-cpp/src/memory_counters.h).

Each engine supports argument `-m` or `--model` with value HuggingFace repository id or local model path.

## Engines

### llama.cpp

https://github.com/ggml-org/llama.cpp

```bash
./engine-llamacpp/run.sh -m "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M" 
```


### MLX

https://github.com/ml-explore/mlx-lm

### MTPLX

https://github.com/youssofal/mtplx

### oMLX

https://github.com/jundot/omlx

### splash

https://github.com/incoai/splash

### uzu

https://github.com/trymirai/uzu
