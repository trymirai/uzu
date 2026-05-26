# M4 cross-simdgroup TG memory contention: a novel optimization finding

**Date:** 2026-05-26
**Hardware:** Apple M4 (GPU family 9)
**Context:** NF4 quantized matmul (QMV) kernel optimization on Apple GPU
**Status:** Validated via 5-run benchmarks; novel relative to published Apple-GPU optimization literature

---

## TL;DR

On Apple M4 GPU, **threadgroup memory regions accessed by multiple simdgroups within a threadgroup incur a hidden cost** that is large (~30 percentage points on quantized matmul kernels) but not directly visible in Apple's exposed GPU counters. The cost is **independent of bank-conflict statistics** and dominates over within-simdgroup bank contention.

The fix: **give each simdgroup exclusive ownership of its own slice of TG memory**, using `simdgroup_barrier` instead of `threadgroup_barrier` for synchronization. We've called this pattern *per-simdgroup partitioning* (vs. lane-level partitioning, which does not capture the benefit).

This pattern does not appear in published Apple GPU optimization material, nor in any major Metal inference engine we've surveyed (llama.cpp, MLX, mps-bitsandbytes), nor in CUDA-targeted LUT-quantization research (FLUTE).

---

## The setup

Kernel: a 16-entry codebook lookup table (LUT) for NF4 weight dequantization, used in the K-loop of QMV with M ∈ {1, 2, 4}. Eight simdgroups per threadgroup, 32 lanes per simdgroup, 256 lanes total. Each K-iteration reads 4 dequantized values from the LUT.

Four variants tested, all with identical K-loop instructions, all bit-exact correctness vs. a reference NF4 dequant kernel:

| Variant | TG memory | Bank coverage | Partition style | Cross-simdgroup data sharing? |
|---|---|---|---|---|
| `Nf4QmvTg` | 32 B (1 LUT) | 8 banks | none — single shared LUT | YES (all 8 simdgroups read same LUT) |
| `Nf4QmvTgReplicated` | 128 B (4 copies) | 32 banks | lane-level (`copy = lane_id >> 3`) | YES (each copy is shared across simdgroups) |
| `Nf4QmvTgVec4` | 128 B (vec4-padded) | 32 banks | lane-level (`replica = lane_id & 3`) | YES (same) |
| **`Nf4QmvTgSimdbar`** | **256 B (8 slices)** | **32 banks** | **simdgroup-level (`slice = simd_group`)** | **NO — exclusive ownership** |

---

## Empirical result

5-run benchmark on M4, batched timing (N=128 dispatches per command buffer, 5 warmup + 20 measured, drop 5 farthest from median). bf16 activations, group_size=64.

### Δ% vs scalar AWQ baseline at M=4, mean ± run-σ

| Shape | `Nf4QmvTg` | `Nf4QmvTgReplicated` | `Nf4QmvTgVec4` | **`Nf4QmvTgSimdbar`** |
|---|---|---|---|---|
| LFM-2048 (2K×2K) | +37.1% ± 0.7 | ~+38% | ~+39% | **+5.1% ± 2.1** |
| Qwen-MLPup (896×4864) | +33.2% ± 3.5 | similar to tg | similar | **+9.2% ± 3.0** |
| Qwen-MLPdown (4864×896) | +46.1% ± 0.5 | ~+47% | ~+49% | **+14.6% ± 3.3** |
| Llama-4096 (4K×4K) | +44.9% ± 1.8 | similar | similar | **+9.7% ± 2.4** |
| Llama-MLPup (4K×14336) | +43.1% ± 5.5 | ~+45% | ~+47% | **+7.3% ± 3.5** |
| Llama-MLPdown (14336×4K) | +48.0% ± 1.0 | ~+49% | ~+49% | **+9.2% ± 1.4** |

Improvement of `Nf4QmvTgSimdbar` over the prior best (`Nf4QmvTg`): **24-39 percentage points at M=4, consistent across all six shapes**. σ ≤ 4pp at all Llama-class shapes.

The improvement is consistent at M=2 (24-34pp) and at M=1 on small shapes (10-30pp); at M=1 on large hidden-dim shapes the kernel is weight-bandwidth-bound and all variants converge.

---

## Why it's not bank conflicts

A naive analysis would attribute the win to better bank coverage: `Nf4QmvTgSimdbar` uses 256 B spanning all 32 banks of TG memory, vs. `Nf4QmvTg`'s 32 B occupying only 8 banks. But:

1. **`Nf4QmvTgReplicated` also achieves 32-bank coverage** (128 B across all banks via lane-partition). It does not capture the win.

2. **At the lane-per-bank-set level, replicated and simdbar are equivalent.** Both have 64 lanes per 8-bank set when all 8 simdgroups execute concurrently — for replicated, because all simdgroups share copies; for simdbar, because two simdgroups (e.g., SG0+SG4) end up wrapping to the same bank range.

3. **The decisive variable is whether different simdgroups touch the same data region.** Lane-level partitioning shares copies across simdgroups; simdgroup-level partitioning does not.

The cross-simdgroup-shared-region cost dominates over within-simdgroup bank conflicts on M4.

---

## Likely mechanism (hypotheses)

We cannot pin down the exact hardware mechanism from Apple's exposed counter set, which lacks per-simdgroup stall reasons and per-line cache coherence telemetry. Plausible candidates, any combination of which could be the actual cause:

### 1. Cache line coherence / read-shared tracking
Lines touched by multiple ALU pipelines may require extra hardware bookkeeping. Apple GPU SMs likely have multiple execution pipelines that can run simdgroups concurrently. A line touched by simdgroups on different pipelines could trigger coherence traffic even for read-only access. Per-simdgroup exclusive ownership eliminates this entirely.

### 2. Memory arbitration at simdgroup granularity
The L1 / TG cache port arbiter may treat each simdgroup's request stream as one client. Eight clients hitting the same cache line have to serialize through the arbiter. Eight clients hitting eight different cache lines can be serviced by parallel ports in one cycle. Lane-level partitioning still has all 8 simdgroups requesting the same lines (just from different lanes); simdgroup-level partitioning has each simdgroup requesting its own line.

### 3. Temporal smearing via `simdgroup_barrier`
`Nf4QmvTgSimdbar` uses `simdgroup_barrier` (32-lane sync) instead of `threadgroup_barrier` (256-lane sync). Even if the barrier wall-clock cost is small (we verified independently — removing the barrier entirely buys only ~7pp), the *behavior* of the barrier matters: with `threadgroup_barrier`, all 256 lanes resume the K-loop in lockstep, hitting the LUT simultaneously. With `simdgroup_barrier`, each simdgroup can drift independently in K-loop progress, smearing accesses across cycles and reducing peak per-cycle contention.

### 4. Hardware cache routing heuristics
Apple GPUs may route private cache lines (touched by one execution unit) differently from shared lines (touched by multiple). Private routing may be faster. Per-simdgroup ownership ensures the "private" routing.

We tested the barrier-only hypothesis by removing the barrier (creating a race) and observed at most ~7pp improvement at the most-suspect cell. The remaining ~25pp of the simdbar win must come from the **structural difference of data ownership**, not the barrier wall-clock cost. The hypotheses above (1-2-4) are consistent with this; (3) is at most a contributing factor.

---

## What does NOT explain the win

Hypotheses ruled out by direct testing:

| Hypothesis | How tested | Result |
|---|---|---|
| Bank conflicts on the 16-entry LUT | `Nf4QmvTgReplicated` (4× copies, 32 banks) | No improvement |
| Bank-spread efficiency at lane granularity | `Nf4QmvTgVec4` (vec4-padded replication) | No improvement |
| `threadgroup_barrier` wall-clock cost | `Nf4QmvTgNoBarrier` (barrier removed, race) | Only ~7pp at one cell |
| Compile-time visibility of codebook | Constant-codebook vs device-buffer variants | Median +0.3pp difference |
| Convert intrinsic cost (`half→float`) | `bfloat2` LUT swap | No measurable change |
| Multi-accumulator ILP within warp | `Nf4QmvTgIlp` (two accumulators) | Null result, slight regression |
| FLUTE-style large LUT duplication | `Nf4QmvByte256Dup` D ∈ {8, 16, 32} | Catastrophic 150-260pp regression |
| Register pressure | Spilled bytes = 0 throughout; reg count 109-110 | Not bottleneck |

The simdgroup-ownership pattern is the only intervention that produced a large win, and it does so cleanly.

---

## Recipe for applying the pattern

For any kernel that uses a small read-shared resource in threadgroup memory:

**Don't:**
```cpp
threadgroup half shared_lut[N];
if (tid < N) shared_lut[tid] = load(tid);
threadgroup_barrier(mem_flags::mem_threadgroup);
// All simdgroups read from shared_lut in K-loop
half v = shared_lut[index];
```

**Do:**
```cpp
threadgroup half partitioned_lut[NUM_SIMDGROUPS * N];
threadgroup half* my_lut = partitioned_lut + simd_group * N;
for (uint i = simd_lane; i < N; i += SIMD_WIDTH) {
    my_lut[i] = load(i);
}
simdgroup_barrier(mem_flags::mem_threadgroup);
// Each simdgroup reads from its own slice
half v = my_lut[index];
```

The cost is `NUM_SIMDGROUPS * N * sizeof(T)` TG memory. For NF4 with 8 simdgroups × 16 halves = 256 B, this is trivial. Even for 256-entry codebooks (NF8), 8 × 256 × 2 = 4 KB, which is within Apple's TG memory budget for typical kernels.

---

## Generalization beyond LUTs

Any small TG resource accessed by all simdgroups is a candidate:

- LUTs / codebooks (NF4, Lloyd-Max, IQ-quantization family, MXFP4)
- Activation function lookup tables (GELU, SiLU approximations)
- Per-group constants (scales, biases) if loaded into TG
- Hadamard rotation factors
- Synchronization flags
- Any data that's "constant per kernel but accessed every iteration"

We have not yet tested the pattern on these other use cases; the prediction is that any kernel currently using a single shared TG resource will see a similar improvement when the resource is partitioned per-simdgroup.

---

## Comparison to published work

Surveyed Metal-related quantization / GPU optimization literature:

- **llama.cpp ggml-metal.metal** — uses `constant float LUT[N]` for IQ4_NL, MXFP4, and similar codebook formats. Loads are served from constant memory (separate hardware path), but the table itself is shared across all threads. No per-simdgroup partitioning.

- **MLX (Apple's ML framework)** — its affine quantization (`affine_qmm_t`) doesn't use codebook LUTs (affine = scale + bias). No relevant comparison.

- **mps-bitsandbytes nf4_matmul.metal** — uses `constant float NF4_CODEBOOK[16]` + tiled matmul. Same pattern as llama.cpp.

- **FLUTE paper (Hong et al., NeurIPS 2024)** — proposes "duplicated vectorized lookup" with D copies of a 256-entry pair LUT. Duplication is lane-level (`lane % D`), not simdgroup-level. CUDA-only target; the bank-conflict story differs from Apple GPU.

- **Hugging Face Metal quantization docs (MLX-based)** — affine 2/4/8-bit, no codebook lookup.

- **Apple Metal Optimization Guide** — discusses bank conflicts in general but does not document the cross-simdgroup contention pattern specifically.

To our knowledge, the per-simdgroup exclusive ownership pattern for shared TG resources on Apple GPU has not been described in any of these sources.

---

## Limitations of this finding

1. **M4-specific empirical signal.** We have not tested on M1, M2, or M3 GPUs. The mechanism likely transfers (same general architecture) but the magnitude may differ.

2. **Counter data does not pinpoint the mechanism.** Apple's exposed counters lack the granularity to distinguish hypotheses 1-4 above. We have strong evidence the pattern works but cannot point at *the* specific microarchitectural cause.

3. **The TG memory overhead scales with simdgroup count.** With 8 simdgroups, overhead is 8×. For larger LUTs (NF8 = 256 entries), this is 4 KB per threadgroup — still within budget on M4 (which has ~32 KB of TG memory per SM) but worth monitoring.

4. **Bench harness lacks thermal monitoring.** Run-to-run σ on M4 may include thermal drift; we don't have automatic cooldown between bench invocations.

5. **Five-run validation, not five hundred.** The signal is robust at 5 runs (σ ≤ 4pp on most cells) but more replications would tighten the estimate further.

---

## Significance

If the finding generalizes:

- A previously-untapped 25-40pp performance lever exists for LUT-quantization kernels on Apple GPU
- The pattern is mechanically simple (per-simdgroup partition + simdgroup_barrier)
- It should apply to all codebook formats (NF4, NF8, IQ-family, Lloyd-Max, MXFP4) and to any small shared TG resource
- It is unrelated to format-specific optimizations (closed-form bypass, etc.) — composes with them
- It is novel relative to surveyed Metal inference engines

Worth: (a) extended testing on M1/M2/M3 GPUs to confirm generality, (b) application to other Metal kernels with shared TG resources (e.g., llama.cpp's IQ-family ports), (c) write-up as a technical report or blog post for the Apple GPU community.

---

## References / artifacts in this repo

- Implementation: `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/nf4_qmv_tg_simdbar.metal` (constant codebook), `nf4_qmv_tg_simdbar_devbuf.metal` (device-buffer codebook, production-flexible)
- Correctness gates: `nf4_tg_simdbar_test.rs`, `nf4_tg_simdbar_devbuf_test.rs`
- Bench: `crates/backend-uzu/tests/performance/matmul/qmm_lut_bench.rs` (qmv_lut_bench)
- GPU traces: `/tmp/qmv_lut_awq.gputrace`, `/tmp/qmv_lut_nf4.gputrace`
- Investigation log: `nf4_lut_investigation.md` (this repo)
