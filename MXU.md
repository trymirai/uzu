# Qwen 27B on M5 Max

Original optimization: Qwen3.6-27B Mirai M4, M5 Max (40 GPU cores), “Tell about London”: 13-token prefill, M=16 speculative decoding. Qwen3.8 profiling is recorded below.

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

Current baseline: `d2e8c57`, Qwen3.8-27B Mirai M4, London, seed 0, temperature 1, top-p .95, top-k 20, thinking/xhigh, no prefix reuse. Five warm baseline requests averaged **91.87 end-to-end tokens/s**: 55 prompt tokens, 283 output tokens, 69 speculative cycles plus prefill. This is a fresh baseline, not a speedup comparison.

| Component | Time |
|---|---:|
| Target + sampling GPU, per cycle | 32.923 ms |
| DFLASH + Weaver GPU, per cycle | 7.225 ms |
| Accepted-state updates GPU | 0.878 ms |
| Full request | 3080.40 ms |
| GPU command-buffer duration sum/request | 2891.37 ms |
| Wall minus GPU sum/request | 189.02 ms (6.14%) |

CPU-only sampling across eight requests attributed approximately 49 ms/request to energy telemetry, 33 ms to resource hazard tracking/barrier encoding, 10.5 ms to temporary access lists, and 2 ms directly to argument-table APIs. These are sampled costs, not additive critical-path savings; some binding work remains unattributed. Metal System Trace added about 17% wall time and was excluded from throughput measurements.

Apple GPU profiler counters from seven isolated baseline shapes found 1.72% occupancy and 2.08% NAX utilization in `BuildTreeGram`. `BuildTreeOut` reported zero GEMM operations on its strict FP32 MPP path; at M=16, three of four SIMD groups have no valid rows. Next target: specialize the DeltaNet tree computation and reduce intermediate work while preserving precision. Counter replay used high GPU state; these percentages do not describe whole-pass utilization or achievable speedup.

All 20 profiling requests matched output and speculative-tree hashes. Results and traces: `target/profiles/forward-breakdown/summary.json` and adjacent files.

DeltaNet follow-up: for BF16 trees of at most 16 tokens on MXU devices, reserve a SIMD group for Gram/inversion and distribute the initial-state projection independently. Fuse the FP32 solve and output products using 16-column SIMD-group tiles, keeping U in registers. Supply the zero state index as a constant. This removes 48 kernel dispatches and 48 GPU fills per Qwen target pass; larger trees retain the existing solve/output path.

| Same London workload, seed 0 | `d2e8c57` | Optimized |
|---|---:|---:|
| End-to-end throughput | 89.25 tokens/s | 90.98 tokens/s |
| Full request | 3170.77 ms | 3110.61 ms |
| Target + sampling GPU/cycle | 33.223 ms | 32.394 ms |

**1.93% end-to-end throughput gain.** Sixteen interleaved pairs, balanced request order, two warmups per version, no prefix reuse. Both versions stayed loaded; GPU requests ran serially. Earlier block runs varied from 83–94 tokens/s and are retained as diagnostics. This is a single-prompt result.

All 36 paired-run requests matched 283 output tokens and 69 speculative-tree hashes. The fused kernel passed 28 bitwise comparisons, including short/ragged shapes; six tree regressions passed with Metal API/shader validation. Temporary helpers, seed/profiling patches and added test cases were removed; the production CLI was rebuilt. Results: `target/profiles/deltanet-tuning/summary.json`.
