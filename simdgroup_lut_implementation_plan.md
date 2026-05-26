# Implementation plan: parameterized simdgroup-partitioned LUT kernel

**Goal:** Build a clean, production-ready, parameterizable LUT quantization kernel for Apple GPU that uses per-simdgroup TG memory partitioning. Branch fresh from `main`, no qmv_lut session debris.

**Scope:** A single kernel family supporting variable bit widths (LUT size scales with bit count). Supports NF4, NF8, Lloyd-Max (2/3/4/6/8-bit), IQ4_NL / MXFP4, and any future codebook format.

---

## Branch setup

```bash
git checkout main
git pull
git checkout -b simdgroup_lut
```

Starting from clean `main`. Do NOT carry over the experimental files from `qmv_lut` branch; this is a from-scratch port that codifies only the validated finding.

---

## Design overview

### Single kernel, parameterized over bit width

```cpp
template <typename T, uint BITS, uint GROUP_SIZE>
VARIANTS(T, bfloat)
VARIANTS(BITS, 2, 4, 8)            // Phase 1: power-of-2 bits only
VARIANTS(GROUP_SIZE, 32, 64, 128)
KERNEL(SimdgroupLutQmv)(
    const device uint32_t* weights,       // packed indices, format depends on BITS
    const device half* codebook,          // (1 << BITS) entries, runtime-set
    const device T* scales,               // per-group scales
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup half codebook_tg[8 * (1 << BITS)],   // per-simdgroup slices
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(32)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(8)
) {
    constexpr uint codebook_size = (1u << BITS);

    // Per-simdgroup exclusive slice (the M4 optimization)
    threadgroup half* my_cb = codebook_tg + simd_group * codebook_size;

    // Cooperative init within this simdgroup only (32 lanes)
    for (uint i = simd_lane; i < codebook_size; i += METAL_SIMD_SIZE) {
        my_cb[i] = codebook[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // K-loop: weight unpacking + LUT lookup + dot product
    // (bit-width-dependent below)
}
```

### TG memory budget per bit width

| BITS | Codebook entries | Bytes per slice | Total TG (8 SG) | Apple M4 budget? |
|---|---|---|---|---|
| 2 | 4 | 8 | 64 B | trivial |
| 3 | 8 | 16 | 128 B | trivial |
| 4 | 16 | 32 | 256 B | trivial |
| 5 | 32 | 64 | 512 B | small |
| 6 | 64 | 128 | 1 KB | small |
| 8 | 256 | 512 | 4 KB | fine — M4 has ~32 KB TG per SM |

All bit widths fit well within Apple M4's TG memory budget.

---

## Phased implementation

### Phase 1: BITS=4 baseline (port the validated win)

**Deliverable:** `SimdgroupLutQmv<T=bfloat, BITS=4, GROUP_SIZE=64>` works, matches the `qmv_lut` branch's `Nf4QmvTgSimdbarDevbuf` performance.

**Steps:**
1. Create `crates/backend-uzu/src/backends/metal/kernel/quant_matmul/simdgroup_lut_qmv.metal`. Port the kernel body from `Nf4QmvTgSimdbarDevbuf` (the device-buffer variant), but generalize the LUT init loop to handle arbitrary codebook size.
2. Weight unpacking for BITS=4: 8 nibbles per `uint32`, standard packing.
3. Correctness gate: `crates/backend-uzu/tests/unit/kernel/quant_matmul/simdgroup_lut_test.rs`. Bit-equivalent to a reference NF4 dequant kernel (within bf16 ULP).
4. Bench column in `crates/backend-uzu/tests/performance/matmul/qmm_lut_bench.rs`.
5. **Validate:** 3-run bench should reproduce ~+7-15% Δ% vs scalar at M=4 across Llama-class shapes (matches what `Nf4QmvTgSimdbarDevbuf` showed on qmv_lut branch).

**Estimated effort:** 1-2 days.

**Files created:**
- `simdgroup_lut_qmv.metal`
- `simdgroup_lut_test.rs`
- Dispatcher entry in `quant_matmul.rs`

---

### Phase 2: BITS=2 and BITS=8

**Deliverable:** `SimdgroupLutQmv` with BITS ∈ {2, 4, 8}.

**Weight unpacking changes per bit width:**
- BITS=2: each `uint32` holds 16 quantized indices (2 bits each)
- BITS=4: 8 indices per `uint32`
- BITS=8: 4 indices per `uint32`

Both 2-bit and 8-bit have clean byte-aligned packing, so unpacking is straightforward.

**For BITS=8 specifically:**
- 256-entry codebook × 8 simdgroups = 4 KB TG memory. Verify M4 doesn't trigger occupancy regression at this footprint. If it does, the existing pattern may not transfer cleanly to 8-bit and we may need a sub-simdgroup partitioning (e.g., share a 256-entry LUT across pairs of simdgroups).

**Validation:**
- BITS=2: bench Lloyd-Max-2bit (per the lalamo codebooks). Expected: bandwidth halves vs BITS=4, but per-K-iter cost similar. Bandwidth-bound regimes should benefit.
- BITS=8: bench against a hypothetical NF8 / IQ8 model. Expected: per-K-iter cost similar, but bandwidth doubles vs BITS=4 — net slower for fixed K.

**Estimated effort:** 2-3 days (different weight-unpack code paths).

---

### Phase 3: BITS=3, 5, 6 (non-byte-aligned packing)

**Hard part:** odd bit widths don't align to bytes, requiring multi-byte shifts across word boundaries during unpack.

**Approach:**
- Define packing format per bit width (probably mirror what Lloyd-Max / lalamo uses)
- Use precomputed shift tables for unpacking
- Or: pack in groups of N quantized values into a fixed number of bytes (e.g., 4 6-bit values into 3 bytes)

**Validation:**
- BITS=6: bench Lloyd-Max-6bit. Per the [[user]] earlier analysis, 6-bit LUT (64 entries × 2B = 128B, all 32 banks) is the theoretical sweet spot for bank coverage. With simdgroup partitioning on top, may close even more of the gap to AWQ.

**Estimated effort:** 3-5 days.

---

### Phase 4: GROUP_SIZE flexibility + Lloyd-Max `bias_lut`

**Deliverable:** Support `GROUP_SIZE ∈ {32, 64, 128}` and the optional per-group bias index used by Lloyd-Max with `bias_bits > 0`.

**Lloyd-Max format primer** (from lalamo PR documentation):
- `packed_weight_indices`: bit-packed indices into the 16-entry codebook
- `packed_scales`: one e4m3 scale per group
- `packed_bias_indices` (optional): bit-packed indices into a per-group bias LUT

Dequant formula: `weight = (codebook[index] - bias_lut[bias_index]) * scale`

**Implementation:** add a second LUT (the bias_lut) that's also per-simdgroup partitioned. Same exclusive-ownership pattern.

**Estimated effort:** 2 days.

---

### Phase 5: QMM-Transposed variant for M ≥ 5

**Deliverable:** `SimdgroupLutQmmTransposed` for the M ≥ 5 GEMM regime.

The existing `QmmTransposed` family uses tiled matmul into TG memory. Applying the simdgroup-partitioning pattern to the dequantized weight tile (Ws) is the natural extension.

**Open question:** does the cross-simdgroup contention pattern manifest in the Ws tile the same way as in the codebook LUT? The Ws tile is accessed per K-iter by the simdgroup_matrix instructions; per-simdgroup partitioning may or may not be applicable to this access pattern.

**Estimated effort:** 3-5 days, includes investigation.

---

## File structure (on simdgroup_lut branch)

```
crates/backend-uzu/src/backends/metal/kernel/quant_matmul/
    simdgroup_lut/
        qmv.metal                      # SimdgroupLutQmv kernel
        qmm_transposed.metal           # GEMM variant (Phase 5)
        helpers.h                      # qdot, unpack helpers per BITS

crates/backend-uzu/src/backends/common/kernel/
    quant_matmul.rs                    # add SimdgroupLut dispatcher entry

crates/backend-uzu/tests/unit/kernel/quant_matmul/
    simdgroup_lut_test.rs              # bit-exactness per BITS
    simdgroup_lut_bias_lut_test.rs     # Lloyd-Max bias LUT (Phase 4)

crates/backend-uzu/tests/performance/matmul/
    simdgroup_lut_bench.rs             # bench harness
    simdgroup_lut_bias_bench.rs        # Lloyd-Max-specific perf

docs/
    simdgroup_lut_design.md            # this plan
    m4_cross_simdgroup_contention.md   # the observation (separate file)
```

---

## API design

### CPU side

```rust
let codebook_buffer = ctx.create_buffer_with_data(NF4_CODEBOOK_BYTES);
encoder.set_buffer(2, &codebook_buffer);
simdgroup_lut_kernel.encode(
    &weights, &codebook_buffer, &scales, &input, &mut output,
    in_vec_size, out_vec_size, batch_size, &mut encoder
);
```

### Model loading

At model load time, the codebook is one of the model's bound buffers (alongside scales, zero-points if any). For format-flexible deployment:
- NF4 model: codebook = the 16 NormalFloat quantile values
- Lloyd-Max model: codebook = the optimization-trained 2^BITS values from the model's metadata CSV
- IQ4_NL model: codebook = the IQ4 quantile values

The kernel doesn't care which — it's just `device half[1 << BITS]`.

### Multiple bit widths in one model

For mixed-precision models (e.g., some layers at NF4, others at NF8), the dispatcher selects the right SPECIALIZE'd kernel instance per layer based on its `BITS` setting.

---

## Bench validation methodology

For each (BITS, GROUP_SIZE) cell:
1. **3-run minimum, 5-run preferred** for reliable run-to-run σ
2. Shapes: LFM-2048, Qwen-MLP variants, Llama-class (4 widths)
3. M values: 1, 2, 4 (decode, light parallel, heavy parallel)
4. Compare against:
   - Scalar AWQ baseline (`QmvFast use_lut=false`) — absolute reference
   - Reference NF dequant kernel of same format (without simdgroup partitioning) — within-format apples-to-apples
5. Report σ explicitly; flag cells where σ > 5% of mean

---

## Optional future work

### Apply the pattern to other shared TG resources

Once the LUT path is validated, audit other Metal kernels for shared TG resources that could benefit:
- Attention's K/V cache tiles?
- Softmax max/sum reduction buffers?
- Hadamard / RHT rotation factors?
- Activation function approximation tables?

Each is a separate experiment but should follow the same pattern.

### Quantification: how does the win scale with bit width?

Hypothesis: the win is larger when the LUT is small (more lookups per byte = more memory pressure per K-iter). Worth testing across BITS=2/4/8 to see if the improvement margin grows or shrinks.

### Profile-guided optimization

Eventually, the runtime could choose simdgroup-partitioned vs. shared-LUT variants based on M (at M=1 large shapes both perform similarly, so the simpler shared-LUT may suffice). This adds dispatcher complexity but could save TG memory in M=1 cases.

### Texture-based LUT as a backup mechanism

If the per-simdgroup pattern hits a footprint ceiling at very large BITS (e.g., BITS=12 with 4096 entries × 8 simdgroups = 64 KB), the texture LUT mechanism we discussed earlier could be a fallback. Different memory hierarchy may avoid the TG-budget constraint.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phase 1 port doesn't reproduce the qmv_lut branch result | Sanity-check on `main` HEAD with bench numbers from `nf4_lut_investigation.md`. If they don't match, the win was an artifact of qmv_lut branch's surrounding state. |
| Non-power-of-2 bit widths are too painful to implement | Ship Phase 1-2 (power-of-2 bits) first as the MVP; defer Phase 3 if needed. NF4 + NF8 covers most use cases. |
| BITS=8 (4 KB TG) regresses due to occupancy | Test occupancy with Xcode Counter view. If it regresses, partition pairs of simdgroups to share LUT (4 slices × 256 entries × 2 = 2 KB). |
| The mechanism doesn't transfer to M1/M2/M3 | Test on at least one other Apple GPU before merging. If it's M4-specific, document carefully and gate the optimization. |
| Bench σ is too noisy to ship-decision | Add `sleep 30` between runs for thermal recovery. Or add automatic `pmset -g therm` monitoring. |

---

## Decision points before starting

1. **Code organization:** new `simdgroup_lut/` subdirectory or flat in `quant_matmul/`? Recommend subdirectory.
2. **Naming:** `SimdgroupLutQmv` or `LutQmv` (the simdgroup pattern becomes the default)? Recommend keeping the descriptive name initially; can rename later when it's the only LUT path.
3. **Existing kernel deprecation:** keep `Nf4QmvConstant` / `Nf4QmvTg` as reference kernels (for sanity / correctness comparison) or delete? Recommend keeping `Nf4QmvConstant` as the "spec reference" and deleting the rest.
4. **Bench harness consolidation:** the `qmv_lut_bench` has accumulated 15+ variant columns this session. New branch should start with a clean 3-4 variant bench (scalar / awq-lut256 / reference NF / simdgroup_lut) and grow only as needed.

---

## Time estimate

- **Phase 1 alone** (BITS=4 port + bench validation on fresh main): **2 days**, shippable as-is for NF4 deployment.
- **Phases 1-2** (BITS=2, 4, 8): **5-7 days**, covers most common use cases.
- **Phases 1-4** (full bit-width range + Lloyd-Max bias_lut): **10-15 days**.
- **Phases 1-5** (including QMM-Transposed): **15-25 days**.

Phase 1 is the minimum viable contribution and stands on its own. Each subsequent phase is independently valuable.
