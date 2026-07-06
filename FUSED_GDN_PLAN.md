# Fused persistent chunk-scan kernel for GDN chunked prefill on Metal

This is the design + execution spec referenced by the `/goal`. Agents: read this
file in full before starting your phase.

## Objective

Implement `DeltaNetChunkedFusedApply`: a single persistent Metal kernel that
replaces the per-chunk serial dispatch chain {`DeltaNetChunkedStateA2Vnew*`,
`DeltaNetChunkedStateA2UpdateDecayScale*`} plus the chunk-parallel
`DeltaNetChunkedOutputAScaledQk*`, eliminating the `h` (~100 MB) and `v_new`
(~100 MB) global temporaries and all per-chunk state DRAM round-trips.
Target: beat the current best chunked path (`ht_full_tile`, ~4.9-5.1 ms) and the
recurrent path (~4.95 ms) at T=4096 on the local machine, with correctness vs
the existing CPU/recurrent references.

## Orchestration rules

- Spawn ALL implementation, debugging, and benchmarking work as subagents via the
  Agent tool with `model: "opus"`. The coordinator does NOT write kernel code, run
  builds, or run benches itself. The coordinator only: reads agent reports,
  analyzes numbers, decides next steps, and dispatches the next agent with a
  precise, self-contained brief.
- Run phases sequentially (implement -> correctness -> bench -> tune). Within a
  phase, parallel agents only for independent work (Metal kernel and CPU mirror
  can be written in parallel).
- Every agent brief must include: file paths, the design-spec section it needs,
  and the exact verification command it must run before reporting success.
- After every code-change session run `cargo fmt` and `./scripts/clang-format.sh`.
- No Co-Authored-By / "Generated with Claude" footers in commits. Do not push
  without being asked. Work on the current branch `chunked_gdn`.

## Background (why)

Repo: uzu, branch `chunked_gdn`. Shapes (Qwen3.5 GDN): num_v_heads HV=48,
num_k_heads HK=16 (GQA 3:1), head_k_dim K=128, head_v_dim V=128, chunk C=64,
T up to 4096 (64 chunks). State S per head: [V=128, K=128] f32.

Current chunked pipeline (all under
`crates/backend-uzu/src/backends/metal/kernel/delta_net/`):
- Chunk-parallel precompute (KEEP UNCHANGED): `chunked_prep.metal`
  (Prep, Cumsum), `chunked_gram.metal` (Gram, ScaleQk), `chunked_solve.metal`
  (Solve), `chunked_build_wu.metal` (BuildW, BuildU), plus
  `DeltaNetChunkedStateA2DecayScale` in `chunked_apply.metal`.
- Serial state chain (REPLACE): the host loops `for chunk_idx in 0..num_chunks`
  dispatching Vnew then Update per chunk (see bench harness at
  `crates/backend-uzu/unit/backends/common/kernel/delta_net_test.rs` around
  lines 1121/1284). Vnew also snapshots state to a bf16 `h` buffer
  [chunks,HV,V,K] solely so a chunk-parallel OutputA can read it later, and
  writes `v_new` f32 [chunks,HV,C,V] read back by Update and OutputA.

At T=4096 this moves ~1.4 GB through DRAM (v_new ~300 MB, h ~200 MB, state
round-trips ~570 MB, W/U ~350 MB) - the pipeline is temporaries-traffic-bound,
not FLOP-bound (~37 GFLOP total). 128 serial dispatch boundaries add ~0.5-1 ms.

Key mathematical property enabling the fix: the chunk recurrence is
embarrassingly parallel over the VALUE dimension. For a value-row slice `vt`:

    Vnew[:, vt] = U[:, vt] - W . S0[vt, :]^T
    Y[:, vt]    = exp(g) (.) (Q . S0[vt, :]^T) + A . Vnew[:, vt]
    S1[vt, :]   = alpha . S0[vt, :] + Vnew[:, vt]^T . B   (B = decay_scale (.) K)

All per-token coefficients (beta, g, alpha=exp(g_last of chunk), decay_scale)
are scalars - no coupling between value slices, ever. So one threadgroup per
(head, value-slice) marches through all chunks with zero global sync and state
never leaving on-chip memory. This matches FLA's CUDA design
(`chunk_gated_delta_rule_fwd_kernel_h`: persistent chunk loop, full K per
program, V blocked) - the one deliberate difference is that we ALSO fuse the
output (OutputA) computation into the scan, because Apple's bandwidth:compute
ratio makes materializing h/v_new the dominant cost, while 192 persistent TGs
already fill 8-40 GPU cores (on NVIDIA the separate chunk-parallel o-kernel is
the right call; on Apple it is not).

## Design spec

New kernel `DeltaNetChunkedFusedApply` (new file `chunked_fused.metal`, follow
the DSL conventions of the existing files: `KERNEL(...)` macro, `VARIANTS`,
`GROUPS`/`THREADS` params, fragment ops from `../matmul/common/fragment.h` +
`simdgroup_fragment_ops.h`; threadgroup memory is declared as kernel params -
see `update.metal` for the pattern).

Dispatch: `hv_idx GROUPS(num_v_heads)` x `v_slice GROUPS(head_v_dim.div_ceil(VT))`,
`THREADS(128)` (4 simdgroups). Template `VT` with `VARIANTS(VT, 16, 32)`;
default/first target VT=32. The kernel loops `c = 0..suffix_len.div_ceil(64)`
internally. `chunk_idx` is NOT a parameter - that's the point.

Inputs: `w`, `u` (bf16, from BuildW/BuildU), `q_norm`, `k_norm` (f32),
`qk_scaled` (f32, this is A), `g` (f32 per-token cumsum), `decay_scale` (f32,
[chunks,HV,C]), `state` (f32, read-write, [HV,V,K]), `out` (model dtype T).
GQA indexing: hk = hv / (HV/HK) for q_norm/k_norm, as in existing kernels.

PRECISION REQUIREMENT (non-negotiable): the state path is f32 end-to-end,
matching the recurrent reference (`prefill.metal` keeps s as float4 registers,
all state operands f32; `update.metal` same - "state stays float"). No bf16
state staging in v1. W/U stay bf16 (value-like inputs, consumed by f32
accumulators). Y intermediate stays f32 until the final `out` cast.

Threadgroup-resident data (VT=32: 24 KB total):
- Canonical state slice stored TRANSPOSED as `S^T [K=128, VT]` f32 (16 KB).
  Stored transposed once so both consumers (W.S^T, Q.S^T) read natural rows.
  `simdgroup_load`/`store` transpose flags / a transposed store handle
  orientation - this dissolves the old h-layout ([V,K] vs [K,V]) problem.
- `Vnew [C=64, VT]` f32 (8 KB).

Per-TG flow: load initial state slice from device (transposed) into TG memory;
then per chunk (3 threadgroup barriers per chunk):
1. Vnew phase - 4 simdgroups x [16, VT] token tiles (the proven FullTile
   shape, cf. `DeltaNetChunkedStateA2VnewFullTile`): acc = U - W.S^T; W
   streamed from device bf16, S^T from TG memory. Store f32 tile to TG Vnew.
   Barrier.
2. Y phase - same [16, VT] tiles: acc = Q.S^T, scale by exp(g_token) via
   map_coords, then acc += A.Vnew (A = qk_scaled streamed from device, Vnew
   from TG). Cast+store straight to `out`. No h, no v_new device traffic.
3. Update phase - each simdgroup owns a [VT, 32] K-column block: init
   register acc elementwise as alpha.S (read TG S^T transposed; alpha = exp(g
   at last VALID token of chunk)), then fragment_mma(acc, Vnew^T (transposed TG
   load), B) where B rows are k_norm scaled by decay_scale (as in
   `DeltaNetChunkedStateA2UpdateDecayScaleFullTile`). Barrier; store acc back
   to TG S^T transposed. Barrier.
After the last chunk, write the final state slice back to the device f32
state buffer (transposed back to [V-rows, K]).

Edge cases: partial last chunk (suffix_len % 64 != 0) - use the existing
`.bounded()` fragment-source patterns and clamp g_last/decay reads to the last
valid token; T < 64 (single partial chunk); verify suffix_len values 1, 63,
64, 65, 200, 256, 4096 in tests.

Also implement a CPU reference mirror module
`crates/backend-uzu/src/backends/cpu/kernel/delta_net/chunked_fused.rs`
following the existing cpu/metal mirror pattern (see how `chunked_apply.rs`
mirrors `chunked_apply.metal`; register in `mod.rs`).

## Phases & acceptance criteria

Phase 1 - Implement (Opus agents; metal kernel + cpu mirror may go in parallel;
also add the Rust kernel wrapper/encode plumbing wherever the existing chunked
kernels are wired - find the pattern with the existing StateA2/OutputA wrappers).
Must compile: `cargo build -p backend-uzu` (or the crate's usual build), plus
formatters.

Phase 2 - Correctness (Opus agent). Add tests in `delta_net_test.rs` following
existing `#[uzu_test]` patterns: fused path vs CPU mirror, AND fused full
pipeline (precompute kernels + fused kernel) vs the recurrent reference
(`DeltaNetPrefill`) at the qwen35 shapes (HV=48, HK=16, K=V=128) across the
suffix_len list above, including a nonzero initial state case. Tolerances: same
or tighter than the existing chunked-vs-recurrent tests (f32 state path should
be tighter than the old bf16-h path). All existing delta_net tests must still
pass. Iterate agent until green.

Phase 3 - Benchmark (Opus agent). Extend/reuse
`bench_delta_net_full_chunked_vs_recurrent_prefill` (run like the existing
ignored benches, e.g.
`cargo test -p backend-uzu --release bench_delta_net_full_chunked_vs_recurrent -- --ignored --nocapture`
- verify exact invocation from the repo). Compare at T in {64, 128, 256, 1024,
4096}: recurrent, current best chunked (ht_full_tile), fused (VT=32 and VT=16).
Use existing warmup conventions (>=500 ms warmup - clock-ramp artifacts
otherwise). Report medians. Success at T=4096: fused < 4.0 ms (stretch: < 3.0
ms). Also report where the recurrent<->fused crossover lands.

Phase 4 - Coordinator analysis. Read the numbers. If fused loses or
underperforms: dispatch a profiling agent (kernel-level breakdown via the
existing bench_delta_net_full_kernel_breakdown pattern) and consider in order:
(a) VT=16 vs 32 occupancy trade, (b) software-pipelining the next chunk's W
load across the update phase, (c) reducing barriers, (d) only as a last resort
and clearly labeled as FLA-equivalent-but-not-recurrent-equivalent numerics: a
bf16 S^T staging VARIANT (FLA itself reads h/v_new as bf16 in its o-kernel).
Do NOT pursue affine P/R transition forms - rejected (2x state FLOPs for
parallelism the persistent grid already provides).

Final report: summary of design-as-built, test status, bench table
(all T, all variants), crossover recommendation for the recurrent/fused switch,
and concrete next steps (e.g. fleet validation on M1-M4, bf16-ifying
q_norm/k_norm/qk_scaled producers to cut another ~150 MB).
