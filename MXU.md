# Qwen 27B on M5 Max

Qwen3.6-27B Mirai M4, M5 Max (40 GPU cores), “Tell about London”: 13-token prefill, M=16 speculative decoding. Qwen3.6 is the shape proxy; Qwen3.8 was not measured.

Profiling identified gate/up (M=16, N=34816, K=5120) as the largest GEMM contributor. Six W4 shapes now use N=32/K=128 tiles, contiguous vector loads over two K=64 groups, and a register output Hadamard transform. Split-K reduction and BF16 rounding are preserved.

M5 Max automatically preprocesses weights into `model_m5_max.safetensors` in the model and speculator folders. Atomic writes, locking and source/layout validation make the cache reusable. Prefill, decode and gathered readout share one packed buffer; input embeddings retain their lookup layout. The original files remain intact. No flag is required.

| Measurement | Original | Cached |
|---|---:|---:|
| Target forward GPU | 35.396 ms | 32.626 ms |
| Complete speculative step | 45.208 ms | 41.977 ms |
| Prompt prefill GPU | 42.172 ms | 41.368 ms |
| Generation | 106.42 tokens/s | 114.61 tokens/s |
| Peak Metal allocation | 16.223 GB | 16.223 GB |

**7.70% complete-step throughput gain**, measured across 12 fixed-seed, 256-token requests in original/cached/cached/original blocks: three requests per block, 318 steps per mode. A step includes draft, target, acceptance and host gaps; it excludes loading, prefill, resets and the final incomplete step. All output hashes matched.

The duplicate-buffer version used 29.703 GB. Cache reuse left file timestamps and inodes unchanged. Validation: 90 bitwise packed-layout comparisons, 68 existing matmul regressions, cache invalidation/concurrency checks, release build and CPU-only check passed. Added test and benchmark code was removed.

Limits: the earlier isolated M=16 hotspot gain was 7.5%, below the 20% target. Three isolated 512-token GEMMs took 7–10% longer; larger-prompt end-to-end performance was not measured.

Raw results: `target/profiles/london-spec/cache-final-metrics.json` and adjacent logs.
