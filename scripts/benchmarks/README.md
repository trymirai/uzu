# Inference benchmarks

Benchmark adapters for inference engines. Each runner loads one model, reads requests from standard input, and writes generated text, timing, throughput, and process memory measurements as JSON Lines. A process can handle multiple requests without reloading its model.

Interaction with every engine follows the same loop: after launch, it waits for a request on stdin, runs inference, writes the result to stdout, and then waits for the next request.

## Engines

### llama.cpp

https://github.com/ggml-org/llama.cpp

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
