# Qwen3.8 S QTIP Metal speed race: report (2026-09-01)

Worktree (private APFS clone of `uzu-qtip-gaussian-metal`, branch `ryan/qtip-gaussian-metal`, base 9022ad3b):

```text
/Users/ryanmathieu/trymirai/uzu-qtip-race-fable
```

Physical package: `/Users/ryanmathieu/trymirai/q38-hyb036-uzu-physical-v1` (re-audited from safetensor spec metadata and code shapes: 166 V4-k2 restarted-64 leaves, 9 V2-k2, 97 V2-k3). G64 control: `performance-benchmarks-m5-20260821/workspace/models/0.15.0/Qwen3.8-27B-M`. Machine: M5 Pro, 20 GPU cores, 64 GB.

## 1. Headline

Whole-model `bench-suffix` (prefix 2048, 2 warm-up / 10 measured forwards, one suffix per process, G64 and QTIP interleaved, run only while no other benchmark process was alive). Final runtime = component-split V4 kernels + device-tensor / transposed kernels + bit-exact transform + repacked head (`results/race_v4_*.json`, `results/race_v5_*.json`):

| Suffix | QTIP race runtime | G64 (same bracket) | QTIP vs G64 | Previous accepted QTIP |
|---:|---:|---:|---:|---:|
| 8 | **123.2 ms** | 234.2 ms | **1.9x faster** | 112.9 ms (numerically wrong, see 3) |
| 16 | 129.5 / 130.3 ms | 95.2 ms | 0.73x | 120.6 ms (numerically wrong) |
| 32 | **137.2 ms** | 119.2 ms | 0.87x | 185.6 ms |
| 64 | **185.2 ms** | 175.8 ms | 0.95x | 259.0 ms |

The machine drifts 5-20% between brackets (thermal state; the other agent's benchmarks also ran concurrently at times), so only within-bracket ratios are meaningful. Peak resident memory: QTIP 9.9-10.6 GB vs G64 17.0-18.2 GB.

Relative to the accepted physical runtime: S32 improved 26%, S64 28%, and S8/S16 are the first *correct* small-suffix numbers. G64 still wins at 16 (by 27%), 32 (13%) and 64 (5%).

In-situ cost split of the QTIP forward at suffix 32 (null-kernel probes, `results/insitu_*_b32.json`, slow machine state): projections 149 ms of 183; transform 4.2 ms; head 10 ms; attention / DeltaNet / norms / embedding / sampling 21.6 ms. The same non-projection budget applies to G64, so G64's GEMMs run in roughly 85-100 ms at suffix 32.

## 2. What changed (all in the private clone)

0. **Component split for V4 leaves** (`qtip_race_v4_cs`, dispatch choices `V4Cs*`): the Q8 V4 table is stored a second time as two 128 KiB `char2` tables (components 0,1 | 2,3) and the A8 activations are permuted once per leaf into two K-packed halves (columns 4g,4g+1 | 4g+2,4g+3; `QtipRacePermuteHalves`). Pass 0 multiplies the (0,1) half with full-density MXU work over K/2 and writes int32 partials; pass 1 does the (2,3) half, adds the partials and applies the epilogue. Each dispatch touches only a 128 KiB table, so every SIMDgroup on the GPU is in the same half at any time. Bit-exact (int32 sums are order independent). Gains at batch 32: mlp_up 1050 → 759 us, in_proj 467 → 349, gate 167 → 143, out_proj 153 → 143; batch 64: mlp_up 1182 → 1104; batch 16: mlp_up 987 → 738. mlp_down (K = 17408) does not gain because each pass re-streams the leaf's 24 MB of codes; a K-split variant that keeps each slab's codes L2-resident (`csk2/4/8`) was slower, so mlp_down stays single-pass.
1. **Projection kernels** (`crates/backend-uzu/src/backends/metal/kernel/matmul/qtip_race.metal`): the signed-A8 activations are consumed by the MXU as a *device tensor operand* (`DeviceTensorOperand<IntegerFormat<8,Signed>>`, the same path the Uzu integer GEMM uses) instead of a per-lane register fragment; V4 restart-state decode is branch-free; table gathers for the next one or two K chunks are issued before the current chunk's MXU work; output is written batch-major for exactly `batch` tokens (no transpose pass at any batch size). One or two row fragments per SIMDgroup, 2/4/8 SIMDgroups per threadgroup. For suffix <= 16 a transposed formulation is used (`qtip_race_b16t`: tokens are the MXU M dimension, weight rows the paired N operand read transposed), which is a legal MXU shape and does no padded work.
2. **Transform** (`qtip_race_transform.metal`): bit-exact replacement of `QtipFullIncoherenceA8` (inputs cached in registers, Walsh-Hadamard strides 1..16 as SIMD shuffles, one threadgroup transpose, strides 32.. as shuffles). Same fp32 operation order, so int8 outputs and scales are bit-identical.
3. **Head**: the i3_s4 readout is repacked at load time into the standard symmetric group-64 4-bit GEMM layout (signed nibbles, bf16 group scale = bf16(row_scale x ladder[index])) and runs the generic Uzu GEMM (`src/encodable_block/embedding.rs`, `repack_i3_s4_to_symmetric_gemm`, env `QTIP_I3_HEAD=i3|u4|u8`, default `u4`).
4. **Dispatch** (`src/backends/metal/kernel/qtip_s_exact.rs`, `select_race_kernel`): table-driven kernel choice per (codec, shape, batch) from the oracle-validated profile; env `QTIP_RACE_PROJ=0` / `QTIP_RACE_TRANSFORM=0` restore the original paths for A/B runs.

Component costs (per forward, harness, one kernel per command buffer, median of 5):

| Component | Original | Race | Suffix |
|---|---:|---:|---:|
| Incoherence transform, all 272 leaves | 10.8 ms / 21.0 ms | 2.9 ms / 4.8 ms | 32 / 64 |
| Head (RHT32 + readout + transpose) | 25.4 ms / 43.0 ms | 6.7 ms / 8.1 ms | 32 / 64 |
| Projection sum, physical kernels vs best correct race kernels | 137 ms / 167 ms | 118 ms / 146 ms | 32 / 64 |
| Projection sum, correct small-suffix kernels | (wrong kernels: 63 / 66 ms) | 107 ms / 107 ms | 8 / 16 |

## 3. Correctness finding: the physical B16 kernels are wrong

`tests/unit/backends/common/kernel/matmul/qtip_race_profile.rs` includes an exact CPU oracle (int32 dot products of the decoded INT8 weights with the signed-A8 activations, then the kernels' fp32 epilogue; enable with `QTIP_RACE_ORACLE=<rows>`). Against it:

- every physical B32/B64 kernel (`QtipGaussianPhysicalQ8*A8Direct{B32,B64}BatchRows`) and every race B32/B64/B16T kernel matches bit-for-bit (one element per few thousand differs by fp32 reassociation in the epilogue);
- the physical B16 kernels (`QtipGaussianPhysicalQ8{V4,V2K2,V2K3}A8DirectB16`, `ROW_FRAGMENTS=2`, MXU row pairing `pair_output_rows` in `fragment_matmul.h`) are wrong for rows 16-31 and 48-63 of every 64-row tile, i.e. half of all outputs, on every family.

The accepted suffix-8 and suffix-16 runtime numbers (112.9 / 120.6 ms) and any suffix-8/16 quality result taken through that path are therefore invalid. Correct 16-token execution needs 32-wide MXU N tiles or the transposed formulation above; the race dispatch never uses the row-paired kernels.

The GEMM head path deviates from the i3 kernel by 3.3e-3 relative RMS on logits (bf16 rounding of the pre-multiplied group scale; identical for the 8-bit variant), with top-1 agreement 30-32/32 on random weights. Projection and transform outputs are bit-exact, so the only numerical difference of the race runtime at suffix >= 32 is the head scale rounding; it must be included in the Roma KL/top-1 gate.

### Whole-model confirmation

`bench-suffix` now also records the greedy-sampled ids of the last measured forward (`sampled_tokens` in the JSON; `results/argmax_*.json`, prefix 2048 of token 1, suffix of token 1):

| Suffix | Original path (`QTIP_RACE_PROJ=0 QTIP_RACE_TRANSFORM=0 QTIP_I3_HEAD=i3`) vs race path | Race with i3 head vs race with u4 head |
|---:|---|---|
| 32 | 32/32 identical ids | 32/32 identical |
| 16 | 0/16: original ids `[95972, 220, 220, 220, ...]` (broken B16 kernels), race ids `[271, 271, ...]` (same as the suffix-32 result) | 16/16 identical |

So at suffix >= 32 the race runtime reproduces the original runtime's greedy decisions exactly, and at suffix <= 16 the original runtime is demonstrably producing wrong logits.

### Rejected in this round (all measured, CPU-oracle checked)

- Activation staging in threadgroup memory (register-path MXU operand, 8/16 SIMDgroups sharing one [tokens x 256] block): slower on V4 and on batch 32; wins only for the two widest V2 shapes at batch 64 (k3 mlp_up 969 → 809 us, k2 mlp_up 932 → 855), where it is used.
- Wide threadgroups (16 / 32 SIMDgroups): no gain; the gather pipeline saturates at 4 threadgroups per core (occupancy probe: per-weight gather cost is flat from 80 to 640 threadgroups).
- Half-table dual dispatch by state parity (predicated gathers): no gain, the instruction count doubles.
- K-split component split: slower.
- Hot-activation diagnostics: MXU-only time is unchanged with hot activations (the MXU phase is compute bound at ~55 TOPS INT8), but gathers + MXU overlap when activation traffic is removed (849 us vs 803 + 213 sequential), i.e. the missing overlap is activation traffic contending with table gathers, which the component split partly relieves.

## 4. Why suffix 16/32/64 still lose

The V4 leaves (166 leaves, 59% of transformer weights) gather `char4` entries from a 256 KiB table. Measured on the largest shape (34816x5120, batch 32): gather+decode only 809 us, MXU only 217 us, full kernel 1050-1080 us, so gathers dominate and do not overlap. Masking states to a 128 KiB / 64 KiB footprint (speed-only diagnostic) drops the full kernel to 546 / 495 us; the V2-k3 kernel (128 KiB table) on the same shape runs at 5 cycles per SIMD gather versus 17 for V4. Deeper prefetch (1, 2, 4 chunks) does not help: the gathers are L1-miss / fill-bandwidth bound, not latency bound. Rejected attempts: split-table (two 128 KiB half tables per K block: 1.5-3x slower, SIMDgroups drift out of phase), half-table dual dispatch with int32 partials (doubled MXU/decode cancels the gain), four row fragments (register pressure). The tables have no exploitable symmetry (checked: no xor-mask symmetry, 65520 distinct Q8 rows).

Instruction-count floor: at the L1-hit rate a 32-lane SIMD gather costs ~4.8 cycles regardless of payload width. The package needs ~270 M gather instructions per forward with the component split (V2: 0.5 per weight, V4: 0.5 per weight after the split), i.e. ~57 ms per forward on 20 cores at 1.5 GHz before any MXU work, so the exact L=16 package cannot beat G64 at suffix 16 and can at best reach parity at 32.

### The lever that changes the floor: an L=15 trellis (same byte format)

For V4 k2 the state is the last 16 bits of the byte stream; with L=15 it is the last 15 bits. The code stream, rate and layout are byte-identical; only the fitter's Viterbi state count (65,536 → 32,768), the table (32,768 x 4 INT8 = 128 KiB) and the decoder mask (0xFFFF → 0x7FFF) change. The `diag_mask15` kernel in this worktree *is* that production kernel (state masked to 15 bits, single pass, one `char4` gather per 4 weights), so its timing is the actual projected cost:

| V4 family (batch 32 / 64) | current best | L=15 table | gain |
|---|---:|---:|---:|
| mlp_up 34816x5120 x37 | 759 / 1118 us | 517 / 875 us | -32% / -22% |
| mlp_down 5120x17408 x61 | 480 / 532 us | 349 / 418 us | -27% / -21% |
| out_proj 5120x6144 x61 | 143 / 171 us | 100 / 122 us | -30% / -29% |
| in_proj 16480x5120 x4 | 397 / 471 us | 320 / 396 us | -19% / -16% |

Projected whole model with only the 166 V4 leaves refit to L=15: S32 ~117 ms (G64 119), S64 ~167 ms (G64 176), S8 ~105 ms (G64 234), S16 ~112 ms (G64 95). Refitting the 106 V2 leaves as V4/L=15 as well (V4/L15 on the 16480x5120 and 8192x5120 shapes measured 271-279 / 121 us at batch 32 vs 296 / 150 for the current k3 kernels) adds roughly another 7 ms at S32 and 5 ms at S64. Suffix 16 stays G64's with any exact-table QTIP representation, because G64's 16-token GEMM path (~60 ms of GEMMs) is below the gather floor.

A V=8 vector width would halve the gather instruction count again, but V8 k2 needs L >= 16 (kV = 16 bits per transition) and a 512 KiB table, which loses the L1.

## 5. How to reproduce

```bash
cd /Users/ryanmathieu/trymirai/uzu-qtip-race-fable
cargo build --release -p cli
RUSTC_BOOTSTRAP=1 cargo test --release -p backend-uzu --no-run
```

Projection profile with the CPU oracle (per family, all variants, batch 8/16/32/64):

```bash
QTIP_RACE_BATCHES=8,16,32,64 QTIP_RACE_ORACLE=128 QTIP_RACE_SKIP_DIAG=1 \
  target/release/deps/backend_uzu-<hash> qtip_race_profile --nocapture
```

Transform bit-exactness, head precision and timing:

```bash
target/release/deps/backend_uzu-<hash> qtip_race_aux_profile --nocapture
```

Whole model (one suffix per process; the script waits for any other `bench-suffix` process):

```bash
bash /private/tmp/claude-501/-Users-ryanmathieu-trymirai/969b20d3-e828-4e80-9b4f-f78e56b3a899/scratchpad/guarded_bracket.sh TAG \
  "g64 <g64 model> 32" "qtip <qtip model> 32"
```

Env switches on the QTIP model: `QTIP_RACE_PROJ=0`, `QTIP_RACE_TRANSFORM=0`, `QTIP_I3_HEAD=i3`.

## 6. Artifacts

```text
results/race_v1_{g64,qtip}_b{32,64}.json     first uncontended bracket (before the small-batch fix)
results/race_v2_*.json                        contended bracket (other agent's benchmarks overlapped; discard)
results/race_v3_{g64,qtip}_b{8,16,32,64}.json guarded bracket, correct kernels, before the component split
results/race_v4_*.json, results/race_v5_*.json guarded brackets with the component split (headline table)
results/insitu_*_b32.json                     null-kernel whole-model cost split
results/argmax_*.json                         whole-model greedy-id equivalence (race == original at suffix 32)
```

Whole-model greedy ids with the final dispatch (component split active) still match the original correct path 32/32 at suffix 32 (`results/argmax_race_b32.json`).

## 7. Round 3 (2026-09-02): the L=15 refit

Ryan's instruction after round 2: "just do the L15 refit and beat it."

### What was refit and how

The shipped Gaussian tables were checked for hidden structure first (65,536 unique rows, no sign or fold symmetry under any bit, so no exact smaller table exists) and the refit was run with the production fitter, unchanged except for the trellis size:

- fitter `sbt_air_biqtip_fit.py` on fat-mirai-4 (arm P8-R0 = V4 k2 restart-64, block LDLQ with the Q3.8 capture Hessians, full-incoherence basis, bf16 gain refit), copied to `sbt_air_biqtip_fit_l15.py` with `QTIP_L=15`;
- the L=15 table is `torch.randn(32768, 4)` from the same seed-1234 CPU generator, which is exactly the first half of the shipped 65,536-row table (verified `torch.equal`); the code stream layout is byte-identical (15-bit seed in the same two bytes, the same 8-bit transitions);
- 166 V4 leaves in three shards on three B200s, 8-38 s per leaf (L=16 took 58-123 s), 25 minutes wall clock; the 106 V2 leaves are untouched (their 128 KiB tables already run at the fast gather rate, and k=3 at V=4 leaves too little state memory);
- composed with `sbt_air_compose.py` at the identical rate 2.398363 bpw; physical streams recovered with a torch-only port of the restarted-stream recovery (`recover_qtip_restarted_stream_l.py --state-bits 15`, dense round trip exact, zero transition violations).

Leaf-level cost of L=15 (fit reports, 166 leaves): median relMSE 0.09330 vs 0.08951 (+4.2%), median Hessian objective ratio 1.045, worst leaf 1.057. The model-level Roma gate is recorded in section 8 when it completes.

### Runtime for L=15

The loader accepts a `[32768, 4]` `codebook_v4` and sets `state_bits = 15` in the kernel arguments; the V4 kernels are the exact single-pass device-tensor kernels with every state masked to 15 bits (`QtipRaceV4L15*`), validated bit-exact against the CPU oracle with `QTIP_RACE_STATE_BITS=15`. Measured picks (min of 7):

| V4 family | batch 16 (padded 32) | batch 32 | batch 64 |
|---|---:|---:|---:|
| mlp_up 34816x5120 x37 | T2Pf2 B16T 497 us | R2Pf2Sg2 511 us | staged-activation Sg16 715 us |
| mlp_down 5120x17408 x61 | Pf2Sg4 342 us | Pf2Sg2 348 us | Pf2Sg4 405 us |
| out_proj 5120x6144 x61 | Pf2Sg4 99 us | Pf2Sg2 100 us | Pf2Sg2 123 us |
| in_proj 16480x5120 x4 | T2Pf2 B16T 177 us | Pf2Sg2 239 us | Pf2Sg2 393 us |
| gate 6144x5120 x3 | Pf2Sg4 101 us | Pf2Sg4 104 us | Pf2Sg2 129 us |

Against the round-2 component-split kernels that is -27..-38% on every V4 shape. Whole model, measured on a package with L=15-shaped tables and the HYB036 codes (identical kernel work to the refit; `results/l15timing_*.json`, clean machine):

| Suffix | QTIP L15 runtime | G64 | tps QTIP / G64 |
|---:|---:|---:|---:|
| 8 | 100.9 ms | 199.7 ms | 79.3 / 40.1 (1.98x) |
| 16 | 108.8 ms | 97.0 ms | 147.0 / 165.0 (0.89x) |
| 32 | 116.1 ms | 115.8 ms | 275.7 / 276.3 (1.00x) |
| 64 | 162.4 ms | 159.7 ms | 394.1 / 400.9 (0.98x) |

### Kernel experiments on top of L=15 (all bit-exact, all rejected)

- Producer/consumer staged weights (`qtip_race_sw`): NP SIMDgroups decode+gather into double-buffered threadgroup tiles while NC SIMDgroups run the MXU. 2-4x slower: 18 KiB of threadgroup memory leaves one threadgroup per core and the producers become latency bound.
- 32/64-token transposed kernels (`qtip_race_bnt`, tokens as the M operand in registers, weights as the paired N operand): ~30% slower than the device-tensor kernels, so the streamed activation operand is not the bottleneck.
- Prefetch depth 1/4 and 8-SIMDgroup threadgroups: within noise of the picks above.
- Staged activations with the L=15 mask: wins only for mlp_up at batch 64 (715 vs 865 us), used there.

The remaining gap at suffix 16 is structural: the gather instruction floor for the 106 V2 leaves alone (two weights per gather) is ~25 ms per forward, and G64's 16-token GEMM path is below the total.

## 8. L=15 quality gate (Roma, fat-mirai-4, `sbt-air-q38-hyb036l15-roma2.log`)

Same manifest, same head and embedding containers, same rate (2.398363 bpw), only the 166 V4 leaves refit at L=15. ROMA-FULL, 37,620 tokens, 100 paired prompts against HYB036:

| metric | HYB036 (L=16) | HYB036-L15 | change |
|---|---:|---:|---:|
| KL | 0.09172 | 0.10191 | +11.1% (bootstrap 95% CI +0.006..+0.014; better on 47/100 prompts, sign p=0.62) |
| TV | 0.07125 | 0.07315 | +2.7% |
| top-1 | 92.96% | 92.63% | -0.33 pt |

The L=15 package sits between HYB036 (KL 0.0917) and the release-tier t0k3full checkpoint (KL 0.1104) on divergence. The team's own finding is that Roma KL correlates positively with MMLU-Pro on this model (better KL, worse task accuracy), so the 12,032-question paired MMLU-Pro is the decisive gate; it needs a ~150 GB GPU and was not run in this session.

## 9. Final result: the L=15 physical package vs G64

Package: `/Users/ryanmathieu/trymirai/q38-hyb036l15-uzu-physical-v1` (8.29 GB, 7.466 GB transformer payload, same 2.3984 bpw; `qtip_shared.codebook_v4` is `[32768, 4]`, 166 V4-T8-R64 leaves refit at L=15, 97 V2-T6 and 9 V2-T4 leaves byte-identical to HYB036). Assets: `q38-hyb036l15-physical-assets-v1` (V2 leaves symlinked to the HYB036 assets, V4 leaves from `sbt-air-q38-hyb036l15-physical-transformer-v1` on fat-mirai-4, every leaf `dense_roundtrip_max_abs` 0.0 and zero transition violations).

Greedy-id sanity on the bench-suffix prompt (both packages on the race runtime): 30/32 agree at suffix 32, 15/16 at suffix 16 (`results/sanity_*.json`; identical weights give 32/32).

Guarded whole-model bracket, clean machine, same-bracket pairs (`results/l15final_*.json`):

| Suffix | QTIP S (L=15 package, race runtime) | G64 M | tps S / M | vs G64 |
|---:|---:|---:|---:|---:|
| 8 | **98.5 ms** | 193.7 ms | 81.2 / 41.3 | **1.97x faster** |
| 16 | 106.2 ms | 91.8 ms | 150.7 / 174.2 | 0.86x |
| 32 | **113.0 ms** | 114.2 ms | 283.1 / 280.3 | **1.01x** |
| 64 | **153.1 ms** | 154.9 ms | 418.1 / 413.1 | **1.01x** |

Peak resident memory 9.8-10.7 GB vs 17.0-18.2 GB. Relative to the accepted physical runtime at the start of the race (185.6 / 259.0 ms at 32 / 64 with the original L=16 package): -39% and -41%.

Honest summary: with the L=15 refit and these kernels, S beats M at suffix 8 by 2x, edges it at 32 and 64 (1% is inside bracket-to-bracket noise, so call it parity-to-slight-win), and still loses at 16 by 14%. Suffix 16 is bounded by the gather-instruction floor of the 106 V2 leaves against G64's cheap 16-token GEMM path; no exact-table QTIP representation measured here closes it. The refit costs +11% Roma KL and -0.33 top-1 points at unchanged rate (section 8); the paired MMLU-Pro gate has not been run.

## 10. Round 4: construction study (held-out, production fitter)

Ryan's ask: explore constructions, QAT/QAD ideas, and test them. Every fit below uses the production
`biqtip_quantize` recipe (block LDLQ, exact Viterbi, restart 64, bf16 gain refit) on a Hessian from 260
capture records and is scored on a disjoint 64-record Hessian (the team's in-sample objectives are
32-45% optimistic on these leaves). Scripts: `l15_table_study*.py` on fat-mirai-4; results in
`l15_table_study.json`, `k3_study.json`, `v4k3_study.json`, `l17_study.json`, `l20_study.json`.

### 10.1 Symmetry-structured tables: L=16 quality in 128 KiB or 64 KiB

Idea: the bitshift trellis only uses the low L-kV bits of a state to choose successors, so the top
state bits are pure history. Define the table on those bits through source symmetries of the Gaussian
residual (negations, a coordinate swap) instead of storing independent rows:
T[s] = sigma(s >> 15) * T15[s & 0x7FFF]. The codeword set is a union of symmetry orbits of a random
base set, so its pairwise-distance statistics match a random codebook of the same size while storing
|base| rows; the successor structure is untouched.

Held-out Hessian objective vs the shipped L=16 full table (two leaves, three more running):

| table | stored | l15/mlp_up | l31/mlp_down |
|---|---:|---:|---:|
| L=16 full (shipped) | 256 KiB | 0 | 0 |
| L=16 antipodal, bit 15 negates the row | 128 KiB | **-0.05%** | **-0.06%** |
| L=16 two sign bits (15: all, 14: comps 0,1) | 64 KiB | **+0.05%** | **+0.03%** |
| L=16 bit 15 negates component 0 only | 128 KiB | +0.66% | +0.66% |
| L=16 four sign bits (12..15 -> comps 0..3) | 16 KiB | +2.58% | +2.53% |
| L=16 four sign bits on the current-symbol bits | 16 KiB | +2.66% | - |
| L=15 plain (the refit shipped in section 9) | 128 KiB | +4.49% | +4.43% |
| L=15 with an EM-trained table (2 iterations) | 128 KiB | +2.89% | - |
| L=14 plain | 64 KiB | +9.81% | - |

So the two structured tables at 128 / 64 KiB are free, and the 16 KiB one costs about half of L=15.
The team's earlier orbit_v4 screen (16 KiB, +17% in-sample) is not reproduced by a clean sign-orbit
construction of the same codeword set; the position of the sign bits (history vs fresh symbol) does not
matter either (+2.58% vs +2.66%).

Kernel cost (min of 9, same run, batch 32 / 64, mlp_up 34816x5120): plain L=15 511 / 842 us,
antipodal 550 / 878, two-sign 542 / 738, four-sign 477 / 646; in_proj 16480x5120: 234 / 369,
259 / 409, 259 / 344, 235 / 315. The 64 KiB table beats L=15 at batch 64 by 8-13% and the 16 KiB table
by 15-24%; the sign select costs ~5% at batch 32.

Two refits are running with these tables (antipodal, two-sign); both are byte-identical L=16 packages
whose table happens to be symmetric, so the loader detects the symmetry (`table_mode` 1/2) and any older
runtime decodes them with the plain 256 KiB path.

### 10.2 k=3 leaves at V=4: the "explodes past 1.0" result was a bug

`sbt_og_qtip._viterbi_block` stores the Viterbi predecessor index in `uint8`; at kV=12 (V=4, k=3) the
index has 4096 values and every path is corrupted, which is why the team recorded k=3 at V=4 as
unusable. With a 16-bit traceback (l3/qkv, held-out, vs the production V2 k=3 leaf at 3.009 bpw):

The production k=3 payloads were fit on all 324 records, so they are in-sample on my 64 held-out
records and score 12.7% (qkv) / 17.9% (gate) better than a fair re-fit on the 260 train records with the
same carried-state fitter. Against the fair V2 k=3 baseline (l3/qkv, held-out):

| construction | physical bpw | held-out vs fair V2 k3 (vs production payload) |
|---|---:|---:|
| V2 k=3 L=16 carried-state re-fit (baseline) | 3.009 | 0 (+12.7%) |
| V2 k=3, two sign bits, 32 KiB table | 3.009 | -1.1% (gate +0.6%) |
| V2 k=3, two signs + swap, 16 KiB table | 3.009 | -1.1% (gate +0.2%) |
| V2 k=3, L=15 plain, 64 KiB | 3.009 | +0.1% |
| V4 k=3, L=16 plain (256 KiB) | 3.125 | **-9.4%** (+2.1%) |
| V4 k=3, L=17, two sign bits (128 KiB) | 3.125 | **-16.9%** (-6.4%) |
| V4 k=3, L=18, three sign bits (128 KiB) | 3.125 | **-22.2%** (-12.4%) |
| V4 k=2 + 4-bit residual VQ | 3.125 | -0.4% (+12%) |
| V4 k=2 + 5-bit residual VQ | 3.25 | -23% (-13%) |

Residual VQ only matches at a higher rate; the V4 k=3 trellis is the real gain and improves with state
memory. The V2 leaves can also take structured tables down to 16 KiB at no cost, and for V2 even L=15
is free (6-bit transitions leave 9-10 bits of history). The rate overhead is the 25-byte restart block (+3.9% bits on k=3 leaves, +1.4% model-wide),
or +1.8% with restart 128. As a speed lever it is not yet a win: the 12-bit decode costs four code-byte
loads per gather, and the exact V4 k=3 kernels written for it (`QtipRaceV4K3Sym20*`) are within +-10% of
the V2 k=3 kernels (batch 64 in_proj -11%, batch 32 qkv +37%).

### 10.3 Restart length vs state bits

L=17 with two sign bits and restart 128 has the same physical rate as L=16 restart 64 (34 bytes per 128
weights) and lands at +4.0%: the free restart every 64 columns is worth more than an extra state bit.

### 10.4 Ideas ranked by evidence (QAT / QAD / construction)

1. **Ship symmetry-structured tables instead of L=15** (tested, free): antipodal 128 KiB or two-sign
   64 KiB, both byte-identical L=16 packages. This removes the +11% KL of the L=15 refit and keeps its
   speed; the 64 KiB variant is 8-13% faster than L=15 at batch 64.
2. **Re-fit the k=3 leaves at V=4 with the fixed Viterbi** (tested at leaf level): -12% held-out
   objective at +3.9% bits on those leaves with an 18-bit state. Quality lever, not a speed lever yet.
3. **Codes-frozen end-to-end fine-tuning of the shared tables** (not tested; the leaf-level EM step is
   its one-leaf proxy and gave -1.6 points on L=15): the tables are 128 K parameters shared by 166 leaves,
   so distilling them against the teacher (AQLM-style codebook fine-tuning) targets model KL directly
   with no runtime or rate change. Same for per-row scales/gains and the norm gammas. This is the highest
   expected-value QAD item because everything it touches is already a continuous parameter in the runtime.
4. **Sequential (quantized-input) Hessians**: the captures are BF16-model activations; feeding each
   layer the activations of the already-quantized prefix (GPTQ/QuIP# style) usually buys several percent
   of KL at 2 bits and needs no new format.
5. **16 KiB four-sign tables** (+2.6% leaf objective, -9% / -24% kernel time at batch 32 / 64): the
   speed/quality knob if suffix 32 must be a clear win; needs its own refit and gate.
6. Rejected by measurement: L=15 with EM tables (+2.9%), restart-128 with more state bits (+4.0%),
   residual VQ for k=3 (+12..25%), L=14 (+9.8%), single-component sign bit (+0.7%).

### 10.5 Calibration volume and damping (l15/mlp_up, antipodal L=16, held-out)

| fit records (tokens/dim) | in-sample objective | held-out objective | relMSE |
|---:|---:|---:|---:|
| 32 (13) | 0.00100 | 0.00469 | 0.1201 |
| 65 (26) | 0.00119 | 0.00390 | 0.1035 |
| 130 (52) | 0.00155 | 0.00328 | 0.0894 |
| 260 (104, production depth) | 0.00191 | 0.00282 | 0.0840 |

Each doubling of calibration still buys 14-17% held-out at the production depth, so the capture set is
not saturated; doubling it is the cheapest quality lever found (forward passes only, no format change).
Damping at 260 records (held-out vs the production 0.01): 0.003 +0.2%, 0.03 -0.5%, 0.1 -1.5%,
0.3 -1.7% (relMSE -12.6%), 1.0 +2.1%. The held-out optimum sits around 0.1-0.3 at this depth: the Hessian
is still over-fit and heavier damping helps, consistent with the team's "damping tracks calibration
depth" note; beyond that the fit collapses toward plain MSE.

## 11. The two-sign package (HYB036-SIGN14): Roma gate

166 V4 leaves refit with the 64 KiB two-sign table (T[s ^ 0x8000] = -T[s], T[s ^ 0x4000] negates
components 0,1), same arm, Hessians, basis and gain refit as HYB036; composed at the identical rate
2.398363 bpw (`sbt-air-q38-base-hyb036sign14`). ROMA-FULL, 100 paired prompts against HYB036
(`sbt-air-q38-hyb036sign14-roma2.log`):

| metric | HYB036 (L=16, 256 KiB) | HYB036-SIGN14 (64 KiB) | change |
|---|---:|---:|---:|
| KL | 0.09172 | **0.07962** | **-13.2%** (better on 64/100 prompts, sign p=0.0066, bootstrap CI -0.0157..-0.0085) |
| TV | 0.07125 | 0.06867 | -3.6% (63/100, p=0.012) |
| top-1 | 92.96% | 93.04% | +0.08 pt |

The leaf-level study says the two-sign table itself is neutral (+0.03..0.05%), so the model-level gain
most likely comes from refitting the 166 leaves with the current capture set and fitter state rather
than from the table; a plain-L=16 control refit would separate the two and was not run. Either way the
package is at least as good as HYB036 and strictly better than the L=15 package (KL 0.1019).

### 10.6 Making the sign select cheap

Applying the symmetry bits as `char4` negations costs far more than the arithmetic suggests: Apple GPUs
emulate 8-bit vector arithmetic through 16-bit conversions, so a single per-row negate took the
gather-bound batch-16 in_proj kernel from 253 to 436 us (+72%) and the two-sign variant to 489 us.
A packed per-byte negation on the 32-bit word (`qtip_race_negate_bytes`: `t = x ^ mask;
inc = ((t & 0x7F7F7F7F) + 0x01010101) ^ (t & 0x80808080)`, merged under the byte mask) does the same
in five integer ops plus one select and brings the overhead down to +3..8% at every batch size. The
kernels remain bit-exact against the oracle. Lesson for any structured-table kernel on this GPU: never
negate int8 vectors component-wise; do it on the packed word.

Same-run timings after the SWAR fix (min of 9, us; the G64 download was streaming so absolute values
are ~5% high, ratios hold):

| shape | batch | L=15 best | antipodal (128 KiB) | two-sign (64 KiB) |
|---|---:|---:|---:|---:|
| mlp_up 34816x5120 | 16 / 32 / 64 | 562 / 522 / 719 | 583 / 534 / 874 | 585 / 573 / 720 |
| mlp_down 5120x17408 | 16 / 32 / 64 | 377 / 344 / 373 | 487 (T2) / 355 / 389 | 398 / 361 / 396 |
| out_proj 5120x6144 | 16 / 32 / 64 | 121 / 99 / 121 | 150 (T2) / 103 / 125 | 128 / 109 / 130 |
| in_proj 16480x5120 | 16 / 32 / 64 | 255 / 354 / 391 | 383 / 368 / 407 | 386 / 323 / 349 |
| gate 6144x5120 | 16 / 32 / 64 | 115 / 123 / 130 | 138 (T2) / 129 / 133 | 129 / 138 / 138 |

The dispatch for two-sign packages therefore mixes: the 64 KiB kernels for in_proj and for mlp_up at batch
64, the antipodal kernels on a half-expanded 128 KiB copy of the table elsewhere. Net cost vs the L=15
package is +1..3 ms per forward at every suffix, for L=16 quality.

## 12. Final bracket: G64 vs the two-sign package vs the L=15 package

The original G64 reference directory was deleted from the Mac during this session (the 470 GB
`performance-benchmarks-m5-20260821` workspace is gone); the reference used below is
`/data/good_models/qwen3.8-27b/qwen3.8-27b-int4-g64-asym` from fat-mirai-2 (uzu package, identical
config.json, identical 18.2 GB footprint; sha256 verified after a parallel byte-range transfer), now at
`/Users/ryanmathieu/trymirai/Qwen3.8-27B-M-g64`. Quiet machine, one process per run, interleaved,
`results/final3_*.json`:

| Suffix | G64 M | HYB036-SIGN14 (L=16 quality, 64 KiB tables) | HYB036-L15 (KL +11%) |
|---:|---:|---:|---:|
| 8 | 188.5 ms (42.4 tps) | **100.4 ms (79.7 tps), 1.88x** | 96.6 ms, 1.95x |
| 16 | 88.9 ms (179.9 tps) | 106.9 ms (149.7 tps), 0.83x | 104.1 ms, 0.85x |
| 32 | 108.5 ms (295.1 tps) | 113.0 ms (283.1 tps), 0.96x | 110.6 ms, 0.98x |
| 64 | 150.7 ms (424.6 tps) | 151.4 ms (422.8 tps), 1.00x | 148.9 ms, 1.01x |

Peak memory 11.5-12.6 GB (SIGN14), 9.8-10.7 GB (L15), 17.0-18.2 GB (G64). Machine state moves the
G64/QTIP ratio by +-3% between brackets (the 01:30 bracket had L15 at 1.01x at suffix 32), so the honest
reading is: 2x at suffix 8, parity within noise at 32 and 64, G64 by 15-17% at 16. The two-sign
package pays ~2% over L=15 for the sign selects and returns L=16 quality (Roma KL 0.0796 vs 0.1019 for
L=15 and 0.0917 for HYB036).

Deliverables: `q38-hyb036sign14-uzu-physical-v1` (recommended: same rate, better Roma than HYB036, L=15
speed), `q38-hyb036l15-uzu-physical-v1` (fastest, KL +11%), runtime in this clone with automatic table
detection (plain / antipodal / two-sign), the held-out study scripts and results on fat-mirai-4.

## 13. Round 5: the final configuration (four-sign V4 + L=15 V2)

Ryan: "go, mog G64, don't stop". Construction chosen from the round-4 ladder: the 166 V4 leaves on the
16 KiB four-sign table (+2.6% leaf objective, kernels -9% / -24% at batch 32 / 64 vs L=15) and the
106 V2 leaves at plain L=15 (free on held-out; L=14 costs +2..4%, two-sign V2 tables cost quality
nothing but their sign ALU makes the V2 kernels slower, so they were dropped). Both refits run on
fat-mirai-4 (`run_v4_sign12_refit.sh`, `run_v2l15_refit.sh`); the combined pipeline `run_final_post.sh`
composes one package (`sbt-air-q38-base-hyb036final`), scores Roma, and recovers all 272 physical streams
(V2 connected streams via `recover_qtip_connected_stream_l.py`, verified byte-exact against the shipped
l3/qkv stream at L=16 before use).

Runtime additions: loader detection of four-sign tables (table_mode 3, 16 KiB stored) and of 32768-row V2
tables (state_bits 15), V2 mask-only L=15 kernels (`QtipRaceK3L15*`, `QtipRaceK2L15*`), the V4 four-sign
batch-16 kernel, and a seed-byte hoist in the V4 gather (-8..-11% at batch 32, free). Picks from a
same-run sweep (min of 9): V4 four-sign batch 16 / 32 / 64: mlp_up 527 / 511 / 686 us, mlp_down
413 / 313 / 389, out_proj 103 / 104 / 122, in_proj 527 / 252 / 327, gate 192 / 108 / 140; V2 k3 L=15:
in_proj 219 / 275 / 379, mlp_up 503 / 525 / 774, qkv 131 / 145 / 197, gate 107 / 114 / 149.

Timing-only package of the final configuration (real tables of the right shape, HYB036 codes) under
user load on the Mac (G64 itself 30% slower than quiet): suffix 32 135.2 ms vs G64 143.7 (-6%), suffix
64 179.5 vs 195.4 (-8%), suffix 16 127.2 vs 117.0 (+9%), suffix 8 119.7 vs 246.8 (2.1x). Quiet-machine
numbers and the real package follow in section 14.

### 13.1 V=8 (one gather per eight weights) is not viable at 2 bits

A V=8, k=2 trellis has 16-bit transitions, so a 20-bit state keeps only 4 bits of history. Held-out on
l47/out_proj vs the shipped V4 L=16: unstructured 2^20-row table (32 MB, the upper bound) +12.6%; 4096-row
base with 8 sign bits (32 KiB) +30.6%; 18-bit state with 6 sign bits +46%. The state memory, not the
table structure, is the limit; L >= 24 would be needed and is out of reach for the fitter. The
shuffle-remap V8 kernel (`QtipRaceV8*`: one 8-byte gather per lane fetches a whole 16x16 fragment, MXU
layout assembled with four SIMD shuffles) was written and compiles but was not pursued further.
Timing bracket of the final-configuration package under moderate load (G64 at 114.7 / 159.3 / 97.5 /
199.1 ms for suffix 32 / 64 / 16 / 8): 110.7 (-3.5%), 156.1 (-2%), 103.5 (+6%), 98.3 (2.0x).

## 14. Final package (HYB036-FINAL: four-sign V4 + L=15 V2)

Roma gate (`sbt-air-q38-hyb036final-roma.log`, same rate 2.398363 bpw, 100 paired prompts vs HYB036):
KL 0.09118 (HYB036 0.09172, -0.6%), TV 0.07424 (0.07125, +4.2%), top-1 92.30 (92.96, -0.66 pt).
So the fastest configuration is HYB036-class on KL and slightly behind on TV/top-1; the two-sign package
(section 11) remains the best quality (KL 0.0796). Whole-model bracket of the real package follows.
### 14.1 Final bracket (quiet machine, `results/finalpkg_*.json`)

Package `/Users/ryanmathieu/trymirai/q38-hyb036final-uzu-physical-v1` (8.29 GB; 166 V4 leaves on the
four-sign 16 KiB table, 97 V2 k3 + 9 V2 k2 leaves at L=15; loader detects both automatically). Greedy
ids vs HYB036: 29/32 at suffix 32, 14/16 at 16. Same-bracket pairs, G64 = the re-fetched int4-g64-asym
package:

| Suffix | G64 M | **FINAL** | two-sign (SIGN14) | L=15 |
|---:|---:|---:|---:|---:|
| 8 | 188.8 ms (42.4 tps) | **94.9 ms (84.3 tps), 1.99x** | 94.4 ms | 91.1 ms |
| 16 | 89.1 ms (179.5 tps) | 99.9 ms (160.1 tps), 0.89x | 101.1 ms | 100.4 ms |
| 32 | 108.7 ms (294.4 tps) | **106.5 ms (300.4 tps), 1.02x** | 109.1 ms | 105.5 ms |
| 64 | 151.2 ms (423.3 tps) | **140.7 ms (455.0 tps), 1.07x** | 147.6 ms | 146.1 ms |

Peak memory 11.4-12.6 GB vs 17.0-18.2 GB. Versus the accepted runtime at the start of the race
(185.6 / 259.0 ms at 32 / 64): -43% and -46%. S beats M at suffix 8 (2x), 32 (2%) and 64 (7%) at
HYB036-class KL; suffix 16 stays G64's by 12%, down from 27% at the start of the race.

## 15. Real speculative decoding (DFlash chain and Weaver tree), 2026-09-03

Setup. One matched Qwen3.8 speculator for every target: DFlash step-6144 (causal-nodecay1d, block 16, target layers 1/16/31/46/61, RoPE 262144) bundled with its Weaver (depth 15), converted with lalamo into uzu's speculator format and symlinked as `<package>/speculator`. Engine knobs added in this clone: `QTIP_SPEC_BATCH` (tree budget, default 16) and `QTIP_SPEC_CHAIN=1` (plain DFlash argmax chain instead of the Weaver tree). The bench now records `tokens_per_forward_pass`. Four prompts (markdown explainer, code, structured list, code review), 256 output tokens, two runs each, quiet machine.

Blocker fixed. The S packages store the LM head as INT3/S4 and the Weaver scores its 512 candidates per node by gathering head rows; uzu had no sparse readout for that format (`sparse i3 readout is not implemented`). Added `QtipI3S4ReadoutSparse{Bf16,F32}` (one simdgroup per candidate, activation row staged once per threadgroup, weights reconstructed exactly as the dense MXU readout does) and wired it through `encode_readout_sparse`. A unit test (`qtip_i3_sparse_readout_matches_dense`) compares it against the dense readout on 2,560 gathered logits: bit-exact (max deviation 0.0, magnitudes up to 130).

Reading the numbers. Real decode throughput = accepted tokens per forward pass / step time; step time here includes the draft (five-layer DFlash over a 16-token block) and, in tree mode, the Weaver rounds, which are identical work for both targets and dilute per-step ratios toward 1.

DFlash argmax chain (no Weaver), pooled (total tokens / total time):

| budget | G64 M | S final | S/M |
|---|---|---|---|
| 8 | 15.9 t/s, 3.52 tok/fwd, 222 ms/step (16 runs) | 27.0 t/s, 3.37 tok/fwd, 125 ms/step | 1.70x |
| 16 | 27.8 t/s, 3.49 tok/fwd, 126 ms/step | 27.0 t/s, 3.42 tok/fwd, 127 ms/step | 0.97x |

Weaver tree, G64 M: 17.2 t/s (3.92 tok/fwd, 229 ms) at budget 8; 38.7 (4.79, 124 ms) at 16; 38.0 (5.38, 142 ms) at 32; 31.3 (5.71, 182 ms) at 64. The tree adds about 26% acceptance over the chain at budget 16 for the same step time.

Acceptance. The S package (KL 0.09) accepts within 1 to 4 percent of the QAT G64 package in chain mode, so the quality gap is not a throughput factor at these budgets.

Weaver tree, pooled over 4 prompts x 2 runs x 256 tokens (micro average: total tokens / total time, total tokens / total forward passes):

| budget | G64 M | S final | S two-sign | S/M (final, two-sign) |
|---|---|---|---|---|
| 8 | 16.4 t/s, 3.72 tok/fwd, 227 ms | 27.6 t/s, 3.70 tok/fwd, 134 ms | 26.9 t/s, 3.61, 134 ms | 1.69x, 1.64x |
| 16 | 35.4 t/s, 4.33 tok/fwd, 122 ms | 31.7 t/s, 4.23 tok/fwd, 134 ms | 33.0 t/s, 4.50, 136 ms | 0.89x, 0.93x |
| 32 | 34.2 t/s, 4.83 tok/fwd, 141 ms | 31.9 t/s, 4.52 tok/fwd, 142 ms | 31.9 t/s, 4.58, 144 ms | 0.93x, 0.93x |
| 64 | 27.7 t/s, 5.01 tok/fwd, 181 ms | 26.5 t/s, 4.61 tok/fwd, 174 ms | 26.4 t/s, 4.80, 181 ms | 0.96x, 0.96x |

Verdict. In the real loop S wins budget 8 by 1.65 to 1.7x and loses 16, 32 and 64 by 4 to 11 percent. Budget 16 is the verify-time gap from the bracket (G64's 16-token GEMM path). Budgets 32 and 64 are different: the S step time is equal or better (174 vs 181 ms at 64) but the S packages accept 6 to 8 percent fewer draft tokens per step, which cancels the narrow verify-time win predicted by the bracket. The chain shows only a 2 to 3 percent acceptance deficit, so the extra loss at wide trees comes through the Weaver: it ranks its 512 candidates per node with the target's LM head, and the S head is INT3 against G64's INT8, so the ranking is noisier exactly where the tree relies on it. Best real throughput per target: G64 35.4 t/s (budget 16), S two-sign 33.0 t/s (budget 16); memory 13 to 18 GB for S against 19 to 23 GB for G64.

Noise. Greedy decoding still varies 5 to 10 percent in accepted tokens between two runs of the same prompt because the Weaver seeds its tree at random, so single-cell ratios within about 3 percent are not decisive; the direction at 16/32/64 is consistent across all four prompts.

## 16. Can a higher-precision head fix the Weaver acceptance loss? (2026-09-03)

Method. `QTIP_WEAVER_HEAD=<package>` loads a second embedding block and hands it to the Weaver's candidate readout only (verification keeps the shipped INT3 head); `QTIP_WEAVER_HEAD_HOT_ROWS=K` scores candidates with id < K on that head and the rest on the shipped head (a merge kernel), which emulates a frequency-tiered head without a refit. The G64 package's head (INT4, 64-column groups, 4.31 bits per weight) served as the higher-precision rows. Pooled over 4 prompts x 2 runs x 256 tokens.

Budget 32, tokens per forward pass:

| Weaver readout rows | tok/fwd | vs INT3 |
|---|---|---|
| G64 M with its own INT4 head (reference) | 4.83 | +6.9% |
| S final, shipped INT3 head | 4.52 | 0 |
| S final, INT4 rows for all ids (readout only) | 4.74 | +4.9% |
| S final, INT4 rows for ids < 65,536 | 4.72 | +4.4% |
| S final, INT4 rows for ids < 32,768 | 4.72 | +4.4% |
| S final, INT4 head for lookup and readout | 4.83 | +6.9% |
| control: S final, own INT3 head through the side path | 4.60 | +1.8% (noise floor) |

Findings. (1) The acceptance deficit at wide trees is the Weaver ranking against the INT3 head: INT4 rows recover most of it and the S lookup embedding is not the cause. (2) The recoverable gain is concentrated in the 32k most frequent token ids (13% of the vocabulary); 90.7% of generated tokens have ids below 32,768 and 97.5% below 65,536. (3) At budget 64 the 64k band measured 4.64 against 4.61 (INT3) and 5.00 (full INT4 head), but budget-64 acceptance varies by up to 8% between greedy runs, so that cell is inconclusive. (4) The side-loaded copy costs 22 to 30 ms per step even when never called (chain mode with the head loaded: 151 vs 129 ms), a residency artifact of the experiment; a shipped tiered head adds about 26 MB and carries none of it.

Bits. Denominator implied by the package (8.293 GB payload at 2.3984 bpw): 27.66B weights. Full INT4 head: 2.456 bpw. Full INT8 head: 2.628 bpw. Tiered head, 32k hot rows at INT4: 2.406 bpw; with rows above 196,608 at INT2: 2.396; with rows above 131,072 at INT2: 2.384. The tiered head is the only variant under the 2.400 cap, and it needs a head refit plus a mixed-format dense readout kernel; the INT2 cold band's effect on verify logits (spurious argmax on rare rows) is unchecked.

Ceiling. Even with G64-level acceptance at every budget, the real-loop picture is: budget 8 S by 1.7x, budget 16 G64 by about 9% (step time, 134 vs 122 ms), budget 32 parity, budget 64 S by about 4%. The head does not touch budget 16.

## 17. Tiered head refit v1 (2026-09-03): built, measured, not shippable as fit

Format. `band_bounds` = [32768, 196608]: rows below 32,768 INT4 on the signed-nibble grid (level = nibble - 8), rows 32,768 to 196,607 the shipped INT3 bytes unchanged, rows from 196,608 INT2 (levels -3, -1, 1, 3). All bands share the row scale x 16-entry ladder group multiplier scheme and the sign-Hadamard-32 input rotation of the shipped head; the loader (`repack_tiered_to_symmetric_gemm`) expands every band onto the symmetric U4 GEMM that verification already uses, and the Weaver readout gathers from that copy through the GEMV path (unit test `qtip_tiered_head_repack_matches_reference` passes; the readout kernel change costs nothing: 141 vs 141 ms at budget 32 on the shipped package). Package `q38-hyb036tiered-uzu-physical-v1`: head payload 475.1 MB (shipped 487.2), 2.3948 bpw with the compose denominator of 26,895,998,464 weights.

Fit. The shipped head's row scales and ladder indices are reproduced exactly by max-over-radius scaling and minimum-error ladder search; only 86% of its codes are nearest-level, and nearest-level refit has lower element error (0.190 vs 0.222), so the shipped middle band was calibrated and is kept byte for byte. Relative reconstruction error in the rotated domain: INT4 hot band 0.103, INT3 middle 0.222, INT2 cold band 0.343 (the 2-bit Gaussian bound, so the cold band is as good as 2 bits get).

Real loop, pooled over 4 prompts x 2 runs:

| budget | G64 M | S shipped head | S tiered v1 | tiered/M |
|---|---|---|---|---|
| tree 8 | 16.4 t/s, 3.72 tok/fwd | 27.6 t/s, 3.70 | 28.4 t/s, 3.77, 133 ms | 1.73x |
| tree 16 | 35.4 t/s, 4.33 | 31.7 t/s, 4.23 | 31.9 t/s, 4.22, 132 ms | 0.90x |
| tree 32 | 34.2 t/s, 4.83 | 31.9 t/s, 4.52 | 33.0 t/s, 4.57, 139 ms | 0.96x |
| tree 64 | 27.7 t/s, 5.01 | 26.5 t/s, 4.61 | 27.7 t/s, 4.74, 171 ms | 1.00x |
| chain 8 | 15.9 t/s, 3.52 | 27.0 t/s, 3.37 | 28.0 t/s, 3.46 | 1.76x |
| chain 16 | 27.8 t/s, 3.49 | 27.0 t/s, 3.42 | 29.9 t/s, 3.67 | 1.08x |

Acceptance moves by +1% (32) to +3% (64) in the tree and +7% in the chain at 16, less than the +4.4% the side experiment got with G64's own INT4 rows for the same 32k band, so nearest-level INT4 rows do not fully reproduce that gain. An 8k hot band gives nothing (4.51 vs 4.52 tok/fwd at budget 32), so the effect needs the 32k band and its 21 MB.

Quality. Roma held-out KL 0.17808 (TV 0.11246, top-1 88.76%) against 0.09118 (0.07424, 92.30%) for the shipped head: the tiered v1 head doubles the KL. The cold band is the multilingual vocabulary (a sample of ids above 196,608: 55% non-ASCII Latin, Cyrillic and Arabic word pieces, 15% CJK and Korean), so 2-bit rows there are unacceptable independent of the KL number. Attribution ablations (INT4 hot band alone; INT2 cold band alone) are recorded below.

Bits. The package has 0.0016 bpw of headroom, worth 8,404 INT4 rows with no compensation, and 8k rows carry no ranking gain. The 32k band costs 21 MB (+0.0072 bpw); paying for it inside the head means INT2 multilingual rows. The remaining option is outside the head: two MLP leaves from V2-k3 to V2-k2 (about 11 MB each) or equivalent, which needs the leaf refit, recompose, Roma and stream-recovery pipeline (hours) and has an unmeasured KL cost.

### 17.1 Attribution and the 32k-hot-only package

Roma held-out ablations (same composed checkpoint, head slices swapped):

| head | KL | TV | top-1 |
|---|---|---|---|
| shipped INT3 | 0.09118 | 0.07424 | 92.30 |
| INT4 hot band (32k rows) only | 0.08993 | 0.07371 | 92.43 |
| INT2 cold band (51.7k rows) only | 0.16945 | 0.10980 | 88.89 |
| both (tiered v1) | 0.17808 | 0.11246 | 88.76 |

The INT4 hot band improves quality; the INT2 multilingual band alone accounts for the whole loss. Package `q38-hyb036hot32k-uzu-physical-v1` (32k INT4 rows, INT3 elsewhere, no INT2): head payload 508.2 MB, 2.4046 bpw (0.0046 over the cap). Real loop, pooled over 4 prompts x 2 runs:

| budget | G64 M | S shipped head | S hot32k | hot32k/M |
|---|---|---|---|---|
| tree 16 | 35.4 t/s, 4.33 tok/fwd, 122 ms | 31.7 t/s, 4.23, 134 ms | 33.6 t/s, 4.34, 129 ms | 0.95x |
| tree 32 | 34.2 t/s, 4.83, 141 ms | 31.9 t/s, 4.52, 142 ms | 34.1 t/s, 4.61, 135 ms | 1.00x |
| tree 64 | 27.7 t/s, 5.01, 181 ms | 26.5 t/s, 4.61, 174 ms | 28.4 t/s, 4.74, 167 ms | 1.02x |
| chain 16 | 27.8 t/s, 3.49 | 27.0 t/s, 3.42 | 30.3 t/s, 3.66, 121 ms | 1.09x |

Budget 8 was not rerun for this package (the head does not enter that result; expect the 1.7x of the other S packages). Its step times run 3 to 5% below the shipped package's across budgets; the readout-kernel control showed no kernel-side difference, so treat that part as run-condition variance rather than a property of the head.

Where this leaves the cap. Hot rows: 8,404 fit with no other change (0.0016 bpw headroom) but 8k rows carry no ranking gain (4.51 vs 4.52 tok/fwd with G64 rows at budget 32); 32k rows carry the gain and cost 0.0072 bpw. Paying inside the head means INT2 multilingual rows (KL x1.9). Paying outside the head (about 21 MB, for example two MLP leaves from V2-k3 to V2-k2) is the remaining route and needs the leaf refit, recompose, Roma and stream-recovery pipeline with an unmeasured KL cost.

8k-row INT4 hot band alone (fits the cap at 2.3999 bpw): KL 0.09002, TV 0.07360, top-1 92.46. The KL gain of the hot band saturates by 8k rows; the acceptance gain does not appear until about 32k rows. So there are two distinct shippable objects: an 8k band that is a free quality improvement under the cap with no speed change, and a 32k band that also buys the acceptance (0.95x/1.00x/1.02x at 16/32/64) at 0.0046 bpw over the cap unless 21 MB is found elsewhere.
