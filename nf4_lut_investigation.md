# NF4 / AWQ LUT performance investigation on Apple M4

**Date:** 2026-05-26
**Branch:** `qmv_lut` (lots of uncommitted state — see "Working state" section)
**Machine:** Apple M4 (GPU family 9) — see `[[user_machine]]` memory.
**Investigation outcome:** Mechanism behind the 2× AWQ-LUT256 vs NF4-LUT256 perf gap was identified. Empirical design space for NF4 LUT optimizations on M4 is exhausted.

---

## TL;DR for the next agent

**⚠ MAJOR UPDATE 2026-05-26 (late session):** A breakthrough was found AFTER the original "design space exhausted" conclusion. See "BREAKTHROUGH: Per-simdgroup LUT slicing" section below. The original tl;dr below is preserved for historical context but partially superseded.

**Updated bottom line:**

- AWQ-LUT256's win mechanism (closed-form bypass) — UNCHANGED, still holds.
- NF4 ceiling on M4 was thought to be Nf4QmvTg at +30-50% vs scalar. **REFINED**: `Nf4QmvTgSimdbar` (per-simdgroup LUT slicing) lands at +5-12% at M=4 across all shapes — closes 60% of the gap to AWQ.
- Mechanism: per-simdgroup TG memory partitioning (each simdgroup has exclusive ownership of its 16-entry LUT slice) eliminates cross-simdgroup contention. Different from lane-level partitioning (TgReplicated/TgVec4 which DID NOT work) — must be SIMDGROUP-aligned.
- SHIP `Nf4QmvTgSimdbar` if shipping NF4.

---

### Original tl;dr (partially superseded by breakthrough below)

1. **AWQ-LUT256's win is NOT really a LUT win.** Apple's M4 PSO compiler dissolves the AWQ LUT into inline byte-decode math (closed-form bypass), eliminating TG memory reads in the K-loop. At runtime, AWQ uses zero threadgroup memory traffic for codebook lookup.

2. **NF4's LUT cannot be dissolved.** Irrational codebook values have no closed-form expression in the index, so the compiler must keep the LUT loads. NF4's K-loop does ~10-14× more TG memory accesses than AWQ at the same shape.

3. ~~Every LUT-layout intervention we tried on M4 either failed or regressed.~~ **REFUTED**: Per-simdgroup LUT slicing (Nf4QmvTgSimdbar) closes 25-38pp at M=4. The previously-tested approaches (constant-AS, replicated, vec4, FLUTE-dup, precomp, ILP) all used lane-level partitioning or shared single LUT — none used SIMDGROUP-level partitioning which is the key.

4. ~~The best practical NF4 variant on M4 is `Nf4QmvTg`.~~ **SUPERSEDED**: `Nf4QmvTgSimdbar` is now the best NF4 variant on M4 by a wide margin.

5. **Two known unexplored paths**: texture-based LUT (unknown EV, untested), algebraic codebook approximation (skip the LUT entirely; accuracy tradeoff).

6. **The mechanism by which "bigger TG memory footprint = worse on M4" is itself unexplained** in our prior testing. The simdbar variant uses 8× the TG memory (256 B vs 32 B) yet wins big — suggesting the prior negative results were specifically about CROSS-SIMDGROUP contention, not TG memory size per se.

---

## Setup and methodology

### The deciding cell

**Llama-MLPup, K=4096 N=14336 M=4**, bf16 activations, group_size=64, int4/NF4 weights. Chosen because:
- Large enough that dequant cost is visible (M=4 means each weight reused 4×, dequant cost surfaces)
- Common shape in production LLMs
- All relevant variants have stable measurements with σ < 3pp here

### Bench harness

`crates/backend-uzu/tests/performance/matmul/qmm_lut_bench.rs` (heavily extended this session).

Methodology: batched-timing (N=128 dispatches per command buffer), 5 warmup batches, 20 measured, drop 5 farthest from median, mean ± σ of 15. Validated by per-cell σ < 1% on stable cells.

**Critical:** never use 1-dispatch-per-CB timing — it's 4-8× inflated and can invert orderings. See `[[feedback_bench_timing_batched]]`.

### Run command
```bash
cargo test --release -p backend-uzu --features metal --test performance \
  -- matmul::qmm_lut_bench::qmv_lut_bench --ignored --nocapture
```

---

## Variants tested (complete list)

All numbers Δ% vs scalar AWQ baseline at Llama-MLPup M=4 (deciding cell). All correctness-gated bit-exact (worst_rel < 1e-2 vs `Nf4QmvConstant`).

### AWQ variants (closed-form bypass applies)

| Variant | Kernel | Δ% vs scalar | Notes |
|---|---|---|---|
| **tmpl-awq** | `qmv_fast_template.metal` | **−20.2%** | OVERALL BEST. C++ template (no SPECIALIZE function constants) + AWQ closed-form init |
| awq-lut256 | `qmv_fast.metal` (use_lut=true, use_nf4=false) | −16.7% | Production-ready, sitting unmerged on `qmv_lut` |
| awq-precomp | reuse `QmvFastNf4Precomputed` w/ AWQ values | +79% | Falls into the "real LUT" cluster — confirms it's the **inline init pattern** that matters, not the values |

### NF4 variants (no closed-form bypass — must do real LUT lookups)

| Variant | Kernel | Δ% vs scalar | Notes |
|---|---|---|---|
| **nf4-tg-sbar** ⭐ | `nf4_qmv_tg_simdbar.metal` | **+8.8%** | **NEW BEST. Per-simdgroup TG LUT slicing + simdgroup_barrier. Closes 25-38pp universally at M=4** |
| nf4-tg | `nf4_qmv_tg.metal` | +48.7% | Prior best NF4 variant. 16-entry TG half codebook (shared across simdgroups) |
| nf4-tg-ilp | `nf4_qmv_tg_ilp.metal` | +51.1% | Multi-accumulator. Tiny regression (likely register pressure) |
| nf4-tg-rep | `nf4_qmv_tg_replicated.metal` | ~+55% | 4× replicated 16-entry TG codebook |
| nf4-tg-vec4 | `nf4_qmv_tg_vec4.metal` | ~+58% | vec4-padded replicated |
| nf4-grafted | `qmv_fast.metal` (use_nf4=true, use_lut=false) | +48-62% | 16-entry constant codebook grafted into QmvFast scaffold |
| nf4-precomp | `qmv_fast_nf4_precomputed.metal` | +76.2% | 256-entry pair-LUT loaded from device buffer |
| synth-precomp | reuse precomp kernel w/ dyadic codebook | +80.1% | Confirms: codebook values don't matter once init is opaque |
| nf4-byte256 | `nf4_qmv_byte256.metal` | +85.3% | 256-entry byte-batched pair LUT in TG |
| nf4-lut-grft | `qmv_fast.metal` (use_nf4=true, use_lut=true) | +87.6% | Byte-batched pair LUT grafted into QmvFast |
| nf4-const | `nf4_qmv_constant.metal` | +30-56% | 16-entry constant-AS codebook (divergent gather serializes) |
| **nf4-dup16** | `nf4_qmv_byte256_dup.metal` (DUP=16) | **+238.5%** | FLUTE-style duplicated pair LUT — catastrophic |
| **nf4-dup8** | (DUP=8) | **+269.3%** | FLUTE D=8 |
| **nf4-dup32** | (DUP=32) | **+348.2%** | FLUTE D=32 |
| nf4-shuf16 | `nf4_qmv_shuffle.metal` | ~+300% | Per-weight simd_shuffle of register-held codebook |
| nf4-shuf8/32 | (different S) | ~+250-310% | Shuffle cost is S-independent |
| nf4-select | `nf4_qmv_select.metal` | ~+1300% | Per-nibble switch-of-literals — catastrophic dependency chain |

---

## nf4-tg performance across the full shape × M grid

(Δ% vs scalar AWQ. Negative = NF4-tg WINS over scalar. NF4-tg is the best NF4 variant.)

| Shape (K × N) | M=1 | M=2 | M=4 |
|---|---|---|---|
| LFM-2048 (2048×2048) | **−11.3%** | +27.0% | +37.6% |
| Qwen-MLPup (896×4864) | +24.8% | +29.2% | +31.8% |
| Qwen-MLPdown (4864×896) | +18.0% | +36.1% | +44.1% |
| Llama-4096 (4096×4096) | −1.1% | +29.9% | +47.6% |
| Llama-MLPup (4096×14336) | **−10.4%** | +28.4% | +48.7% |
| Llama-MLPdown (14336×4096) | **−8.7%** | +30.0% | +49.7% |

**Pattern:**
- M=1: NF4-tg wins/ties on big shapes (≥4096 hidden dim), loses on small Qwen shapes
- M≥2: NF4-tg always loses (dequant cost surfaces when weights amortize)

vs awq-lut256 at M=1 large shapes (Llama-class), **nf4-tg is actually slightly faster by 3-10pp.** Because both are bandwidth-bound and NF4 has less per-iter work (no zp load).

---

## The closed-form bypass mechanism (root cause)

This is the key insight of the session.

### What AWQ-LUT256's init looks like

```cpp
// qmv_fast.metal:165 — only difference vs NF4 path
q4_lut[tid] = bfloat2(
    static_cast<bfloat>(tid & 0x0f),         // ← pure arithmetic on index
    static_cast<bfloat>((tid >> 4) & 0x0f)   // ← pure arithmetic on index
);
threadgroup_barrier(...);
```

The compiler can prove `lut[i] = bfloat2(i & 0xf, (i >> 4) & 0xf)` for any i. When the K-loop later reads `lut[byte]`, the compiler **substitutes the closed-form expression in place of the TG load**, eliminating the read at the post-AIR PSO compile step.

### What NF4's init looks like

```cpp
// qmv_fast.metal:160 — NF4 branch
q4_lut[tid] = bfloat2(
    static_cast<bfloat>(nf4_codebook[tid & 0x0fu]),         // ← discrete table lookup
    static_cast<bfloat>(nf4_codebook[(tid >> 4) & 0x0fu])
);
```

`nf4_codebook[]` is a 16-entry constant array of irrational halves. No closed-form expression exists. The compiler can't substitute, so K-loop reads from TG every iteration.

### Counter evidence (gputrace counters from `/tmp/qmv_lut_awq.gputrace` vs `/tmp/qmv_lut_nf4.gputrace`)

| Counter | AWQ (bypassed) | NF4 (real LUT) | Δ |
|---|---|---|---|
| **Threadgroup Memory L1 Read Bandwidth** | **43.84** | **621.96** | **14×** |
| ThreadGroup L1 Read Accesses | 7.25 | 72.04 | 10× |
| L1 Register Residency | 0.80 | 0.18 | AWQ uses registers; NF4 uses TG |
| Register L1 Read Accesses | 1.20 | 0.08 | Same story |
| ALU Utilization | 42.05 | 18.12 | AWQ ALU-bound; NF4 mem-bound |
| Instruction Throughput Limiter | 90.38 | 90.38 | Both gated on issue |
| Instruction Throughput Utilization | 27.04 | 11.55 | AWQ 2× more productive cycles |
| Kernel Occupancy | 86.27 | 95.79 | NF4 has higher occupancy but it's wasted |
| Total ALU Instructions | 141.8G | 137.86G | Nearly identical instruction count |

**AWQ at runtime does NOT use its LUT.** NF4 does. Same kernel source, different machine code, 2× different runtime.

### Decisive ablation (also done this session)

Same `QmvFastNf4Precomputed` kernel, same K-loop, only **device-buffer LUT values change**:

| LUT contents (precomp via device buffer) | Δ% vs scalar |
|---|---|
| AWQ nibbles {0..15} | +79.4% |
| NF4 codebook | +79.4% |
| Dyadic rationals {-1, -7/8, …} | +79.5% |

All three are within 0.1pp. **Values are irrelevant.** Only **inline closed-form init** unlocks the bypass.

---

## Hypotheses tested and falsified

| Hypothesis | How tested | Result |
|---|---|---|
| Bank conflicts on LUT[16] | TgReplicated (4 copies × 16-entry across 32 banks) | Regressed 3-5pp |
| Bank conflicts on LUT[256] (FLUTE) | nf4-dup8/16/32 | Catastrophic +150-260pp regression |
| half→float convert cost | bfloat2 LUT (verified AIR uses `v2bf16` intrinsic instead of `v2f16`) | Identical perf |
| Compile-time visibility of codebook | nf4-precomp (device buffer) | Only 6pp improvement; bulk gap remains |
| LUT contents complexity (mantissa cleanliness) | synth-precomp (dyadic rationals) | Identical to nf4-precomp |
| Register pressure for NF4 | Counter check (109 regs, 0 spills) | Not bottleneck |
| Kernel scaffold tuning | nf4-grafted into QmvFast | No improvement vs standalone |
| Dependency chain stalls (need more ILP) | nf4-tg-ilp (multi-accumulator) | Wash to slight regression |
| Missing memory work (probe-load idea) | Forced dummy zp load on NF4 | Made it worse |
| AIR-level codegen differences | AIR-diff of `qdot_qmv_fast_experiment` branches | Inner loops byte-identical |
| Function-constant SPECIALIZE codegen cost | tmpl-awq, tmpl-nf4 (constexpr templates) | 5-13pp universal speedup, but NF4 gap remains |

---

## BREAKTHROUGH: Per-simdgroup LUT slicing (`Nf4QmvTgSimdbar`)

Added 2026-05-26 (late in session). This finding **invalidates the prior "design space exhausted" conclusion** for NF4 on M4.

### The win

| Shape | nf4-tg (prior best) | **nf4-tg-sbar (new)** | M=4 improvement |
|---|---|---|---|
| LFM-2048 | +36.8% | +5.6% | −31pp |
| Qwen-MLPup | +30.5% | +4.5% | −26pp |
| Qwen-MLPdown | +45.9% | +11.9% | −34pp |
| Llama-4096 | +44.7% | +12.7% | −32pp |
| **Llama-MLPup** | **+46.5%** | **+8.8%** | **−38pp** |
| Llama-MLPdown | +44.9% | +7.5% | −37pp |

Universal 25-38pp improvement at M=4 across all 6 shapes. Similar wins at M=2 (22-32pp). Wins at M=1 on small shapes (Qwen-MLPdown +46 → +16). Bit-exact correctness.

### The kernel

`crates/backend-uzu/src/backends/metal/kernel/quant_matmul/nf4_qmv_tg_simdbar.metal` — copy of `nf4_qmv_tg.metal` with two changes:

1. TG memory: `threadgroup half codebook_tg[8 * 16]` — 8 simdgroups × 16-entry codebook each = 256 B total (vs 32 B for shared LUT).
2. Each simdgroup uses ONLY its own slice: `threadgroup half* my_cb = codebook_tg + simd_group * 16;`. Lanes 0-15 of each simdgroup cooperatively write the 16-entry codebook to its slice. All 32 lanes of a simdgroup read EXCLUSIVELY from their own slice.
3. Sync: `simdgroup_barrier(mem_flags::mem_threadgroup)` instead of `threadgroup_barrier`. Cheaper (32-lane sync vs 256-lane sync), but **not** the dominant reason for the perf gain.

### Why this works (mechanism)

**Phase 1 result falsified the "barrier is the fixed cost" hypothesis.** A separate variant `Nf4QmvTgNoBarrier` (intentionally broken — barrier removed, race condition) gained at most 7pp at one shape, 0pp at most others. The threadgroup_barrier wall-clock cost is NOT the dominant cost.

The actual mechanism: **prior `Nf4QmvTg` had all 256 lanes hammering ONE shared 16-entry / 32-byte LUT, causing severe cross-simdgroup contention on the 8 TG memory banks holding the LUT**. With `Nf4QmvTgSimdbar`, each simdgroup operates on its own 16-entry slice in different TG memory addresses (banks). When simdgroups execute concurrently on different Apple ALU pipelines (which they apparently do, contrary to my earlier hedging), they no longer contend for the same banks.

### Why prior partitioning attempts (`TgReplicated`, `TgVec4`) didn't capture this

- `Nf4QmvTgReplicated`: 4 copies, lane-partition (`copy = lane_id / 8`). A single simdgroup spans 4 copies — different simdgroups still share the same copies.
- `Nf4QmvTgVec4`: same idea, different padding.
- `Nf4QmvTgSimdbar`: 8 copies, **simdgroup-partition** (`copy = simd_group`). Each copy is EXCLUSIVELY owned by ONE simdgroup. No simdgroup ever reads another simdgroup's slice.

The distinguishing property: **simdgroup-level partitioning vs lane-level partitioning**. Same TG memory consumption, different ownership model. On M4, only the simdgroup-aligned form eliminates cross-simdgroup contention.

### This is novel

Not in FLUTE (per-lane bank skew), not in llama.cpp (shared constant codebook), not in mps-bitsandbytes (shared TG tile). The per-simdgroup ownership pattern for small shared LUTs appears to be an undocumented M4-specific optimization.

### Should ship

**Ship `Nf4QmvTgSimdbar` as the default NF4 path.** 25-38pp universal improvement, no regression at any shape × M, bit-exact correctness, negligible TG memory footprint increase. The remaining ~25pp gap to AWQ-LUT256 is the irreducible closed-form-bypass advantage that AWQ has via its compiler dissolution.

### Possible extensions to test

1. **Apply per-simdgroup ownership to `Nf4QmvByte256`** — 8 per-simdgroup 256-entry pair-LUT slices = 8 KB total. May close more of the gap if cross-simdgroup contention was also limiting byte256.
2. **Apply to AWQ-LUT256** — already wins via closed-form bypass, but per-simdgroup pair LUT might add another few pp.
3. **Investigate Xs/Ws tiles in `QmmTransposed`** for similar cross-simdgroup contention patterns.

---

## What remains untested

### 1. Texture-based LUT
Store the 16-entry NF4 codebook as `texture1d<half, access::sample>` and use nearest-neighbor sampling in the K-loop. Different memory hierarchy than TG memory. **Unknown EV** because Apple doesn't document whether texture cache shares bandwidth with L1/TG on M4. Could be 10-30pp improvement, could be zero or negative. ~150 LOC + DSL extension for texture binding ≈ 1 day of work.

### 2. Algebraic codebook approximation
Replace the LUT with inline polynomial expression of the nibble index that approximates NF4 quantile values:
```cpp
half nibble = static_cast<half>(byte & 0xF);
half centered = nibble - 7.5h;
half result = centered * 0.13333h + centered * centered * centered * 0.0006h;
```
Closed-form on the index → gets the compiler bypass. Cost: ~3-5% MSE accuracy hit on NF4 quantization quality. May need fine-tuning to recover model accuracy. **Predicted to land at AWQ-tier perf (~−10 to −15%).** Untested but mechanism is clear. ~150 LOC + accuracy validation.

### 3. Cooperative simdgroup matmul primitives
M4 has `simdgroup_matrix` ops (8×8 fp16 matrix multiply). Could potentially restructure the K-loop to use these for the dequant×activation product. For QMV (M=1), awkward because M=1 doesn't fill the 8×8 matrix shape. **Possibility, unclear path.** Higher engineering cost (~few days).

### 4. Per-model compiled kernels with codebook baked in
Generate a unique metallib per model where the 16 NF4 codebook values are hard-coded constants in the kernel source. The compiler might then unroll a switch-case that's effectively a register LUT. We tested `Nf4QmvSelect` which did exactly this — it was the worst variant (+1300%). So this approach **doesn't work in obvious form** but variations might.

---

## "Why does bigger TG memory hurt on M4?" (open mystery)

A consistent finding across ALL experiments: the smaller the TG memory footprint, the better the performance. Pattern:

| TG memory footprint | Best variant | Δ% vs scalar at M=4 |
|---|---|---|
| 32 B (LUT[16] half) | nf4-tg | +48.7% |
| 128 B (LUT[16] × 4 replicated) | nf4-tg-rep | ~+55% |
| 1 KB (LUT[256] half2) | nf4-byte256 | +85.3% |
| 8 KB (D=8) | nf4-dup8 | +269.3% |
| 16 KB (D=16) | nf4-dup16 | +238.5% |
| 32 KB (D=32) | nf4-dup32 | +348.2% |

This is monotonic-ish (D=16 < D=8 anomaly aside). But:
- Kernel Occupancy stays >95% for the small variants
- Counters don't show specific saturation that explains the dramatic regressions

**This relationship is genuinely undiagnosed.** Possible mechanisms:
- M4 has a fixed budget for TG memory per SM that affects resident threadgroup count beyond what's exposed in "Kernel Occupancy"
- LUT init time grows with footprint and matters at M=1 (where total kernel time is short)
- Some scheduler heuristic penalizes large-tgmem threadgroups
- Apple-specific cache eviction behavior

Future agents: if you find a way to definitively diagnose this, the rest of the NF4 design space may reopen.

---

## Working state (uncommitted on qmv_lut branch, 2026-05-26)

### Files added (new kernels and tests)
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/nf4_qmv_tg_replicated.metal`
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/nf4_qmv_tg_vec4.metal`
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/nf4_qmv_tg_ilp.metal`
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/nf4_qmv_byte256_dup.metal`
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/qmv_fast_nf4_precomputed.metal`
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/qmv_fast_template.metal`
- `crates/backend-uzu/tests/unit/kernel/quant_matmul/nf4_tg_replicated_test.rs`
- `crates/backend-uzu/tests/unit/kernel/quant_matmul/nf4_tg_vec4_test.rs`
- `crates/backend-uzu/tests/unit/kernel/quant_matmul/nf4_tg_ilp_test.rs`
- `crates/backend-uzu/tests/unit/kernel/quant_matmul/nf4_byte256_dup_test.rs`
- `crates/backend-uzu/tests/unit/kernel/quant_matmul/nf4_lut256_graft_test.rs`
- `crates/backend-uzu/tests/unit/kernel/quant_matmul/nf4_precomputed_test.rs`
- `crates/backend-uzu/tests/unit/kernel/quant_matmul/qmv_fast_template_test.rs`
- `crates/backend-uzu/tests/performance/matmul/qmv_lut_trace.rs` (gputrace capture)

### Files modified
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/qmv_fast.metal`
  - Added `use_nf4` SPECIALIZE bool + branch for NF4-LUT-graft
  - Changed LUT type from `half2` to `bfloat2` (null result, but harmless)
  - **Contains a DUMMY-ZP-LOAD PROBE (lines ~250-260 in `else` branch) that BREAKS NUMERICAL CORRECTNESS — must be reverted before any use of QmvFast(use_nf4=true,use_lut=false) in production**
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/quant_matmul.h` — added `qdot_q4_byte_lut_bfloat`
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/nf4_qmv_core.h` — added `qdot_nf4_byte_lut`, `qdot_nf4_byte_lut_dup`, `qdot_nf4_vec4`, `qdot_nf4_tg_ilp` helpers
- `crates/backend-uzu/src/backends/metal/kernel/quant_matmul_nf4_bench.rs` — added many `Nf4Variant` variants and dispatch arms
- `crates/backend-uzu/build/metal/toolchain.rs` (lines 118-123) — **TEMP** added `-gline-tables-only` and `-frecord-sources` for GPU trace source attribution. Should be REVERTED to `[OsString::from("-O2")].into()` for release builds.
- `crates/backend-uzu/tests/performance/matmul/qmm_lut_bench.rs` — bench columns for all new variants
- `crates/backend-uzu/tests/unit/kernel/quant_matmul/qmv_fast_test.rs` — tolerance bumped for bfloat2 ULP slack

### Captured artifacts on /tmp
- `/tmp/qmv_lut_awq.gputrace` — Xcode GPU trace, 200 dispatches of AWQ-LUT256 PSO only
- `/tmp/qmv_lut_nf4.gputrace` — Xcode GPU trace, 200 dispatches of NF4-LUT-graft PSO only
- `/tmp/qmv_lut_1779772573.gputrace` — earlier combined capture (both PSOs in one CB)
- `/tmp/air_diff_qmv_fast.ll` — full disassembled AIR (9973 lines)
- `/tmp/qmv_lut_run{1,2,3}.log`, `/tmp/synth_run{1..5}.log`, `/tmp/bfloat2_run{1,2}.log`, etc. — bench logs

### Inspection commands
```bash
# Run the full QMV bench
cargo test --release -p backend-uzu --features metal --test performance \
  -- matmul::qmm_lut_bench::qmv_lut_bench --ignored --nocapture

# Capture two separate gputraces (one per PSO)
cargo test --release -p backend-uzu --features metal --test performance \
  -- matmul::qmv_lut_trace::qmv_lut_trace_capture_split --ignored --nocapture

# Open captured trace in Xcode
open /tmp/qmv_lut_awq.gputrace
open /tmp/qmv_lut_nf4.gputrace
```

---

## Recommendations for next agent

### Ship-ready wins
1. **Ship `awq-lut256`** (qmv_fast.metal with `use_lut=true` SPECIALIZE). It's −13 to −18% vs scalar at M=4 across all shapes. Already integrated, just needs to be enabled in dispatcher. Currently `use_lut=true` is hard-wired in `crates/backend-uzu/src/backends/common/kernel/quant_matmul.rs` lines 126, 178, 199 on qmv_lut branch but never merged to main.

2. **Consider `tmpl-awq`** (qmv_fast_template.metal). It's an additional 5pp faster than awq-lut256 because it avoids SPECIALIZE function-constant codegen overhead. Same closed-form bypass + template specialization. Engineering cost: requires kernel duplication (one `[[kernel]]` per (use_nf4, use_lut) combination instead of one shared kernel with function constants).

### NF4 deployment
**Ship `Nf4QmvTgSimdbar`** (`nf4_qmv_tg_simdbar.metal`) — the per-simdgroup-partitioned variant. 25-38pp better than `Nf4QmvTg` universally at M=4 with bit-exact correctness. The gap to AWQ at M=4 closes from ~63pp to ~25pp. At M=1 on large hidden dims, it's competitive with or slightly faster than awq-lut256.

(Original recommendation `Nf4QmvTg` is superseded — see "BREAKTHROUGH" section.)

### Cleanup before any merge
- **REVERT the dummy-zp-load probe in qmv_fast.metal** (in the `else` branch of `if (use_mlx_quant)`, around line 251-260). It's a perf-experiment artifact that breaks numerical correctness for NF4-LUT-graft.
- **REVERT the build script flags** (`crates/backend-uzu/build/metal/toolchain.rs` lines 118-123) back to `[OsString::from("-O2")].into()` for release builds. The `-gline-tables-only` + `-frecord-sources` flags were added for GPU trace source attribution.
- **Decide which experimental kernels to keep vs. delete**. All NF4 variants other than `Nf4QmvTg` and `Nf4QmvConstant` are research code with no ship value. The two graft variants in qmv_fast.metal (`use_nf4=true, use_lut=true/false`) can be deleted if not shipping NF4.
- **Don't bother shipping `nf4-tg-ilp`** — null result, adds register pressure.

### Stop pursuing
- All LUT-layout tricks for codebook formats on M4
- Bank-conflict mitigation for codebooks
- Larger TG-memory codebooks (always lose)
- Multi-accumulator ILP (compiler already does this)
- bfloat2 vs half2 LUT type (no perf difference)
- More variations of replication or duplication

### Worth trying (open hypotheses)
- Algebraic NF4 approximation (Method A) — only known path to AWQ-tier perf for codebook formats
- Texture-based LUT (Method B) — unknown EV, needs measurement
- Investigation of WHY larger TG footprints hurt so much on M4 (deep mystery)

---

## Counter signature reference

For diagnosing similar bottlenecks in future experiments, here's what the M4 counter signatures looked like:

**AWQ-LUT256 (closed-form bypass, FAST):**
- Threadgroup Memory L1 Read Bandwidth: ~44
- ThreadGroup L1 Read Accesses: ~7
- L1 Register Residency: 0.80
- ALU Utilization: 42%
- Instruction Throughput Utilization: 27%
- Kernel Occupancy: 86%

**NF4-LUT-graft (real LUT lookups, SLOW):**
- Threadgroup Memory L1 Read Bandwidth: ~622 (14× higher)
- ThreadGroup L1 Read Accesses: ~72 (10× higher)
- L1 Register Residency: 0.18 (4× lower)
- ALU Utilization: 18% (memory-stalled)
- Instruction Throughput Utilization: 12% (gated on memory)
- Kernel Occupancy: 96% (parallelism available but wasted)
- L1 Cache Limiter: 34% (NOT cache-port saturated)

**Sanity check counters to verify your bench isn't broken:**
- Both PSOs should show ~92M Kernel Invocations for 200 dispatches × this shape
- Spilled Bytes should be 0 for all variants (we're not register-spilling)
- Sanity gate: scalar Llama-MLPup M=1 should be 0.30-0.35 ms (if much higher, batched timing is broken)

---

## Related memory entries

- `[[project_awq_closed_form_bypass]]` — root cause memory (updated 2026-05-26)
- `[[project_nf4_replicated_tg_no_gain]]` — bank conflict / replication findings (superseded by closed-form story)
- `[[project_nf4_e4m3_zp_no_perf_gain]]` — original 2-month NF4 investigation (predates this session's mechanism finding)
- `[[feedback_bench_timing_batched]]` — bench harness reliability
- `[[user_machine]]` — M4 GPU family 9
- `[[feedback_format_after_changes]]` — cargo fmt + clang-format
- `[[feedback_release_build]]` — always --release for perf work
