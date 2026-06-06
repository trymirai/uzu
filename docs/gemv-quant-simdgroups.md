# Quantized GEMV: core-count-aware simdgroup selection

## Summary

Quantized GEMV (matrix-vector / small-batch matmul, `M ≤ 4`) is the hot path of
autoregressive decode. On Apple GPUs with many cores, uzu's quant GEMV was
leaving decode throughput on the table for **narrow-output, long-reduction**
shapes (small `N`, large `K`) because each threadgroup used 8 simdgroups, which
launches too few threadgroups to fill the GPU.

The fix (`crates/backend-uzu/src/backends/metal/kernel/matmul/gemv/kernel.rs`,
`GemvSpecialization::select`) switches such dispatches to **2 simdgroups per
threadgroup** — 4× more threadgroups — when, and only when, it helps:

```
use 2 simdgroups  ⇔  quant && !RHT
                     && k >= QUANT_LARGE_K            (8192)
                     && m * ceil(n / 32) < cores * MIN_SG8_TGS_PER_CORE   (8)
```

i.e. when an 8-simdgroup launch would produce fewer than 8 threadgroups per GPU
core *and* the reduction is long enough to amortize 4× more threadgroup
launches. Everything else — full precision, RHT, low-core GPUs, large `N`, short
`K`, and `M ≥ 2` (already enough threadgroups) — keeps the original 8-simdgroup
kernel. No Metal shader changes were needed: the 2-simdgroup variants already
existed; only the host dispatch heuristic changed.

`MetalContext::gpu_core_count()` was exposed and threaded into `select`. An env
override `UZU_GEMV_QUANT_SIMDGROUPS=2|8` forces the choice for tuning.

## Result (end-to-end, M4 Max, 40 GPU cores)

Qwen3.5 uzu (this branch) vs MLX, `qwen35_uzu_vs_mlx_fast` profile, decode
tokens/s (ratio > 1.0 = uzu faster):

| model | quant | uzu dec t/s | mlx dec t/s | uzu/mlx | uzu prefill t/s | mlx prefill t/s |
|-------|-------|------------:|------------:|:-------:|----------------:|----------------:|
| 0.8B  | int4  | 524.8 | 464.1 | **1.13** | 8649 | 4555 |
| 0.8B  | int8  | 390.1 | 349.8 | **1.12** | 8674 | 5305 |
| 2B    | int4  | 311.2 | 288.9 | **1.08** | 3954 | 3220 |
| 2B    | int8  | 201.3 | 183.8 | **1.10** | 3991 | 3180 |
| 4B    | int4  | 149.3 | 143.1 | **1.04** | 1579 | 1440 |
| 4B    | int8  |  89.5 |  88.9 | **1.01** | 1581 | 1415 |

uzu beats MLX on decode across all six model/quant pairs (largest margin on the
smaller models, where decode is most GEMV-bound), and ~1.5–1.9× on prefill.

## How the research got here

### 1. A head-to-head GEMV microbenchmark

Added `pmetal-mlx-rs` (the maintained `mlx-rs` fork) as a macOS dev-dependency
and `gemv_vs_mlx_bench.rs`: the same FP-bf16 and 4-bit-affine shapes through
uzu's GEMV kernels and MLX's `matmul` / `quantized_matmul`, measured
symmetrically (a fixed `CHUNK` of GEMVs per GPU submission so per-op kernel
throughput dominates submission overhead).

It showed uzu at parity-or-better on most shapes, but **trailing MLX on the
`M=1` quant decode case at small `N`** — e.g. `M1·K14336·N4096` (≈19% slower).

### 2. The measurement trap: thermal noise

On M2 Max, GPU kernel time drifts **5–20% run-to-run** from clock/thermal
state. A naive before/after `cargo bench` comparison was dominated by this — it
produced a fake "win" and even showed unchanged code as 25% different. Only two
measurements are trustworthy:

- **Clock-controlled GPU traces** via `tools/gpu_trace` (xctrace + hardware
  counters), confirming both A and B sat at `Maximum` performance state.
- **Same-build env-toggle A/B** (`UZU_GEMV_QUANT_SIMDGROUPS`) run back-to-back.

### 3. Diagnosis: occupancy-bound, not bandwidth-bound

GPU-counter traces of the losing shape (`M1·K14336·N4096`, 4-bit, at Maximum
clock) vs a winning shape:

| shape | Compute Occupancy | Buffer-Read Limiter | interpretation |
|-------|:-----------------:|:-------------------:|----------------|
| N=4096 (uzu loses) | 17% | 66% | memory **not** saturated |
| N=14336 (uzu wins) | 23% | 89% | ~memory-bound (optimal) |

At small `N`, only `ceil(4096/32)=128` threadgroups launch — too few to fill a
38–40-core GPU and hide memory latency, so neither ALU nor bandwidth saturates.
The lever is more threadgroups. Quant GEMV is K_SPLIT-locked, so the only knob
is simdgroups-per-threadgroup; dropping 8→2 quadruples the threadgroup count
(matching MLX's qmv granularity).

Clock-controlled trace of the fix on the target shape: compute occupancy
17%→22%, total GEMV GPU time **730→582 ms for an identical dispatch count (≈20%
faster)**.

### 4. Cross-chip validation (SSH to M1…M4-Max) — the win is core-count-dependent

The fix was tuned on a 38-core M2 Max. Running the same sg8-vs-sg2 A/B on every
chip (self-contained `gemv_profile` example over SSH) showed the win only exists
on high-core GPUs:

| chip | GPU cores | sg2 / sg8 (M1·K14336·N4096) | verdict |
|------|:---------:|:--------------------------:|---------|
| M1      | 8  | ~1.01 | tie (already filled) |
| M2 / M4 | 10 | ~1.00 | tie |
| M2 Pro  | 19 | 0.94  | sg2 −6% |
| M4 Pro  | 20 | 0.98  | sg2 −2% |
| M2 Max  | 38 | ~0.91 | sg2 −9% |
| M4 Max  | 40 | 0.86  | sg2 −14% |

On ≤10-core parts, 128 threadgroups already fill the GPU, so 2 simdgroups only
add launch overhead. And `M=4` (512 threadgroups) is already filled even on big
GPUs, so it must stay at 8. Hence the gate on `m * ceil(n/32) < cores * 8`,
which is core-count-aware and naturally excludes `M ≥ 2`. Re-validated with the
auto-heuristic on every chip: **zero regression anywhere** (worst +0.8% noise),
boost preserved on high-core.

### 5. The `k >= 8192` gate

Without it, the heuristic would also fire on short reductions (e.g.
`K4096·N4096`), where the kernel is too short to amortize 4× more threadgroup
launches — sg2 was up to **~1.7× slower** there (M2 Max trace). Tuned on
`K ∈ {4096, 14336}`; intermediate `K` should be validated on the target device
(the env override makes that easy).

### 6. Experiments tried and reverted (rigor)

- **Widening 4-bit weight loads** in `qdot` (`ushort`→`uint`): the AIR showed
  narrow loads, but a clock-controlled A/B showed a **uniform 7–28% regression**
  — the hardware already coalesces contiguous narrow loads. Reverted.
- **`K_SPLIT` for small-N FP GEMV**: inconclusive (within the thermal noise
  floor; FP small-N is already at ~peak bandwidth). Reverted.

## Tooling added

- `gemv_vs_mlx_bench.rs` — uzu vs MLX GEMV comparison (macOS, `pmetal-mlx-rs`).
- `fp_gemv_bench.rs` — FP GEMV `gpu_execution_time` bench (the stable oracle).
- `examples/gemv_profile.rs` — criterion-free harness for `tools/gpu_trace`
  (sustains max clock; honors `UZU_GEMV_QUANT_SIMDGROUPS`).

## Reproduce

```sh
# microbench (macOS): uzu vs MLX
cargo bench -p backend-uzu --bench kernel -- 'Mlx/Kernel/Gemv'

# clock-controlled A/B of the fix (any Apple GPU)
gpu-trace run --gpu-counters -- target/release/examples/gemv_profile quant 1 14336 4096 64 4 120 8   # sg8
gpu-trace run --gpu-counters -- target/release/examples/gemv_profile quant 1 14336 4096 64 4 120 2   # sg2

# end-to-end model comparison
uv run benchmarks run <id> --task london-summary --profile qwen35_uzu_vs_mlx_fast --uzu-branch <branch>
```
