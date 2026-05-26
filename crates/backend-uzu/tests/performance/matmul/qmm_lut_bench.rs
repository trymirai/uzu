#![cfg(metal_backend)]

//! 3-way batched-timing bench: scalar vs awq-lut256 vs nf4.
//!
//! ROOT-CAUSE TIMING FIX: encode N≈128 dispatches of the SAME kernel into
//! ONE command buffer, submit once, `wait_until_completed`,
//! `gpu_execution_time() / N` = per-dispatch time. One-dispatch-per-CB is
//! swamped by fixed CB overhead (4-8x inflation + inverted ordering).
//!
//! THREE kernels per shape/M (all gs=64, bf16, no bias):
//!  - scalar      (BASELINE): QmvFast / QmmTransposed with use_lut=false
//!                 (int4 mantissa-trick scalar dequant), use_zero_points=true,
//!                 use_mlx_quant=false, packed 4-bit zero-points + bf16 scale.
//!  - awq-lut256:  the SAME kernel + SAME inputs, use_lut=true (256-entry
//!                 threadgroup LUT dequant).
//!  - nf4:         Nf4QmvConstant / Nf4QmmConstant (16-entry NormalFloat
//!                 codebook, bf16 per-group scale).
//!
//! FORMAT CONFOUND: scalar/awq-lut256 are uniform asymmetric int4; nf4 uses
//! the fixed 16-entry NF codebook. This is a matched-shape kernel-cost
//! comparison, NOT an isoaccuracy comparison.
//!
//! Reducer (user spec): 5 warmup batched-submits, 20 measured; from the 20
//! per-dispatch values drop the 5 farthest from the median; mean ± σ of the
//! kept 15.

use backend_uzu::{
    DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{QuantizedMatmulQmmTransposedKernel, QuantizedMatmulQmvFastKernel},
        },
        metal::{
            Metal,
            kernel::quant_matmul_nf4_bench::{Nf4QmmBench, Nf4QmmTile, Nf4QmvBench, Nf4Variant},
        },
    },
};
use half::{bf16, f16};

use crate::common::helpers::alloc_buffer_with_data;

type Ctx = <Metal as Backend>::Context;
type B = Metal;

const GROUP_SIZE: usize = 64;
const BITS: usize = 4;

// Batched timing parameters (user spec).
const WARMUP_BATCHES: usize = 5;
const MEASURE_BATCHES: usize = 20;
const DROP: usize = 5; // drop the 5 farthest-from-median of the 20
// Dispatches encoded into one command buffer (amortizes fixed CB overhead).
const N_DISPATCH: usize = 128;

/// Mirror of the production AWQ dispatcher's tile pick:
///   batch_dim < 48 -> small (BM=8/BK=32/BN=32/WM=WN=1)
///   batch_dim >= 48 -> big  (BM=64/BK=64/BN=64/WM=WN=2)
fn pick_tile(m: usize) -> (u32, u32, u32, u32, u32, Nf4QmmTile, &'static str) {
    if m < 48 {
        (8, 32, 32, 1, 1, Nf4QmmTile::Small, "BM=8")
    } else {
        (64, 64, 64, 2, 2, Nf4QmmTile::Big, "BM=64")
    }
}

fn bf16_buf(
    ctx: &Ctx,
    values: &[f32],
) -> <B as Backend>::DenseBuffer {
    let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    alloc_buffer_with_data::<B, bf16>(ctx, &data)
}

/// Pack raw 4-bit weight nibbles into u32 words (8 nibbles per word).
/// This is what QmvFast / QmmTransposed expect (vs NF4 which takes raw u8).
fn pack_weights_u32(
    values: &[u8],
    bits: usize,
) -> Vec<u32> {
    assert_eq!(bits, 4);
    values
        .chunks(8)
        .map(|chunk| {
            let mut word = 0u32;
            for (i, &v) in chunk.iter().enumerate() {
                word |= ((v & 0xF) as u32) << (i * 4);
            }
            word
        })
        .collect()
}

/// Pack a row of 4-bit zero-point indices two-per-byte (matches the unit
/// test's `pack_zero_points` for bits==4).
fn pack_zp_row(values: &[u8]) -> Vec<u8> {
    values
        .chunks(2)
        .map(|chunk| {
            let lo = chunk[0] & 0x0F;
            let hi = if chunk.len() > 1 {
                chunk[1] & 0x0F
            } else {
                0
            };
            lo | (hi << 4)
        })
        .collect()
}

#[derive(Clone)]
struct Stats {
    mean: f64,
    std: f64,
}

/// User reducer: from the per-dispatch values, drop `DROP` farthest from the
/// median, arithmetic mean ± σ of the kept set.
fn reduce(per_dispatch_ms: &[f64]) -> Stats {
    let mut sorted = per_dispatch_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let mut by_dist: Vec<f64> = per_dispatch_ms.to_vec();
    by_dist.sort_by(|a, b| (a - median).abs().partial_cmp(&(b - median).abs()).unwrap());
    let keep = &by_dist[..by_dist.len().saturating_sub(DROP)];
    let mean = keep.iter().sum::<f64>() / keep.len() as f64;
    let var = keep.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / keep.len() as f64;
    Stats {
        mean,
        std: var.sqrt(),
    }
}

/// Run the batched-timing protocol for a single `encode_one` closure.
/// `encode_one(&mut encoder)` encodes exactly ONE dispatch of the kernel.
fn time_batched<F>(
    ctx: &Ctx,
    mut encode_one: F,
) -> Stats
where
    F: FnMut(&mut Encoder<B>),
{
    let one_submit = |encode_one: &mut F| -> f64 {
        let mut encoder = Encoder::new(ctx).unwrap();
        for _ in 0..N_DISPATCH {
            encode_one(&mut encoder);
        }
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        let total_ms = completed.gpu_execution_time().as_secs_f64() * 1000.0;
        total_ms / N_DISPATCH as f64
    };
    for _ in 0..WARMUP_BATCHES {
        let _ = one_submit(&mut encode_one);
    }
    let mut per_dispatch: Vec<f64> = Vec::with_capacity(MEASURE_BATCHES);
    for _ in 0..MEASURE_BATCHES {
        per_dispatch.push(one_submit(&mut encode_one));
    }
    reduce(&per_dispatch)
}

fn dpct(
    base: &Stats,
    chal: &Stats,
) -> f64 {
    (chal.mean - base.mean) / base.mean * 100.0
}

// ============================ shared buffers ============================

struct CellBuffers {
    // scalar / awq-lut256 (QmvFast / QmmTransposed) inputs:
    w_u32: <B as Backend>::DenseBuffer, // u32-packed 4-bit weights
    s_bf16: <B as Backend>::DenseBuffer,
    zp_scalar: <B as Backend>::DenseBuffer, // QmvFast-layout packed zero points
    // nf4 inputs:
    w_u8: <B as Backend>::DenseBuffer, // raw u8-packed 4-bit weights
    x_buf: <B as Backend>::DenseBuffer,
    y_buf: <B as Backend>::DenseBuffer,
    /// 256-entry bfloat2 LUT (= 512 bf16) precomputed CPU-side from the NF4
    /// codebook. Bound by `QmvFastNf4Precomputed` only.
    nf4_precomputed_lut: <B as Backend>::DenseBuffer,
    /// 256-entry bfloat2 LUT precomputed from the SYNTHETIC 16-entry
    /// dyadic-rational codebook (multiples of 1/8 in [-1, +7/8]). Ablation
    /// to isolate the "irrational mantissa" hypothesis from operand range.
    /// Same kernel as `nf4_precomputed_lut` — only the 512 bf16 contents
    /// differ.
    synth_precomputed_lut: <B as Backend>::DenseBuffer,
    /// 256-entry bfloat2 LUT precomputed from the AWQ nibble codebook
    /// {0.0, 1.0, ..., 15.0}. Same kernel as `nf4_precomputed_lut` — only
    /// the 512 bf16 contents differ. Tests whether the closed-form bypass
    /// requires INLINE init code or just CLEAN VALUES from any source.
    awq_precomputed_lut: <B as Backend>::DenseBuffer,
    /// 16-entry `half` codebook buffer for `Nf4QmvTgSimdbarDevbuf`. Populated
    /// CPU-side from `NF4_CODEBOOK`. In production this would be a per-model
    /// resource; for the bench it's allocated once per cell.
    nf4_codebook_buf: <B as Backend>::DenseBuffer,
}

/// CPU-side NF4 16-entry codebook. MUST stay byte-identical to the
/// `nf4_codebook[]` half literals in `nf4_common.h` (modulo half→bf16 round).
const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.5250730,
    -0.39491748,
    -0.28444138,
    -0.18477343,
    -0.09105003,
    0.0,
    0.07958029,
    0.16093750,
    0.24611230,
    0.33791524,
    0.44070983,
    0.56261432,
    0.72295684,
    1.0,
];

/// Byte → (bfloat2(low_nibble, high_nibble)) expansion = 256 entries × 2 bf16
/// = 512 bf16 values. Flat layout: [b0.lo, b0.hi, b1.lo, b1.hi, ...].
fn precompute_nf4_byte_lut() -> Vec<bf16> {
    let mut out = Vec::with_capacity(512);
    for b in 0u32..256 {
        out.push(bf16::from_f32(NF4_CODEBOOK[(b & 0xF) as usize]));
        out.push(bf16::from_f32(NF4_CODEBOOK[((b >> 4) & 0xF) as usize]));
    }
    out
}

/// Synthetic 16-entry codebook: multiples of 1/8 in [-1, +7/8]. Same range as
/// NF4 (~[-1, 1]) but all entries are exact dyadic rationals (mantissas have
/// few bits set in fp16/bf16). Ablation lever for the "operand mantissa
/// cleanliness" hypothesis: kernel binary and TG-memory layout are identical
/// to the NF4 precomputed-LUT path; only the 256 bfloat2 values differ.
const SYNTH_CODEBOOK: [f32; 16] = [
    -1.0,
    -7.0 / 8.0,
    -6.0 / 8.0,
    -5.0 / 8.0,
    -4.0 / 8.0,
    -3.0 / 8.0,
    -2.0 / 8.0,
    -1.0 / 8.0,
    0.0,
    1.0 / 8.0,
    2.0 / 8.0,
    3.0 / 8.0,
    4.0 / 8.0,
    5.0 / 8.0,
    6.0 / 8.0,
    7.0 / 8.0,
];

/// Same byte → bfloat2 expansion as `precompute_nf4_byte_lut`, but populated
/// from `SYNTH_CODEBOOK` instead of `NF4_CODEBOOK`.
fn precompute_synth_byte_lut() -> Vec<bf16> {
    let mut out = Vec::with_capacity(512);
    for b in 0u32..256 {
        out.push(bf16::from_f32(SYNTH_CODEBOOK[(b & 0xF) as usize]));
        out.push(bf16::from_f32(SYNTH_CODEBOOK[((b >> 4) & 0xF) as usize]));
    }
    out
}

/// AWQ nibble codebook: integers 0..15 as f32. Matches the values the inline
/// `q4_lut[i] = bfloat2(i & 0xf, ...)` init produces in the awq-lut256 path.
const AWQ_NIBBLE_CODEBOOK: [f32; 16] =
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];

/// Same byte → bfloat2 expansion as `precompute_nf4_byte_lut`, but populated
/// from `AWQ_NIBBLE_CODEBOOK`. Tests whether closed-form bypass requires INLINE
/// source or just clean operand values.
fn precompute_awq_byte_lut() -> Vec<bf16> {
    let mut out = Vec::with_capacity(512);
    for b in 0u32..256 {
        out.push(bf16::from_f32(AWQ_NIBBLE_CODEBOOK[(b & 0xF) as usize]));
        out.push(bf16::from_f32(AWQ_NIBBLE_CODEBOOK[((b >> 4) & 0xF) as usize]));
    }
    out
}

fn build_buffers(
    ctx: &Ctx,
    input_dim: usize,
    output_dim: usize,
    m: usize,
) -> CellBuffers {
    let num_groups = input_dim / GROUP_SIZE;
    // Raw 4-bit weight nibbles (one per K element per output row).
    let weights_raw: Vec<u8> = (0..(output_dim * input_dim)).map(|i| ((i * 7 + 1) % 16) as u8).collect();
    let w_u32_data = pack_weights_u32(&weights_raw, BITS);
    // NF4 raw u8 packed (2 nibbles per byte) layout.
    let packed_bytes_per_row = (input_dim * BITS) / 8;
    let weights_u8: Vec<u8> = (0..(output_dim * packed_bytes_per_row)).map(|i| (i % 251) as u8).collect();

    let scales_f32: Vec<f32> = (0..(output_dim * num_groups)).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();

    // scalar/awq zero points: QmvFast/QmmTransposed expect, per output row,
    // `zp_stride = (num_groups+1)/2` bytes (4-bit indices two-per-byte).
    let zp_stride = (num_groups + 1) / 2;
    let mut zp_scalar_packed: Vec<u8> = Vec::with_capacity(output_dim * zp_stride);
    for j in 0..output_dim {
        let row: Vec<u8> = (0..num_groups).map(|g| ((j * 2 + g * 3) % 16) as u8).collect();
        let mut packed = pack_zp_row(&row);
        packed.resize(zp_stride, 0);
        zp_scalar_packed.extend_from_slice(&packed);
    }

    let x_f32: Vec<f32> = (0..(m * input_dim)).map(|i| ((i % 257) as f32) / 257.0).collect();

    let nf4_lut_bf16 = precompute_nf4_byte_lut();
    let synth_lut_bf16 = precompute_synth_byte_lut();
    let awq_lut_bf16 = precompute_awq_byte_lut();
    let nf4_codebook_half: Vec<f16> = NF4_CODEBOOK.iter().map(|&v| f16::from_f32(v)).collect();

    CellBuffers {
        w_u32: alloc_buffer_with_data::<B, u32>(ctx, &w_u32_data),
        s_bf16: bf16_buf(ctx, &scales_f32),
        zp_scalar: alloc_buffer_with_data::<B, u8>(ctx, &zp_scalar_packed),
        w_u8: alloc_buffer_with_data::<B, u8>(ctx, &weights_u8),
        x_buf: bf16_buf(ctx, &x_f32),
        y_buf: ctx.create_buffer(m * output_dim * DataType::BF16.size_in_bytes()).expect("y buf"),
        nf4_precomputed_lut: alloc_buffer_with_data::<B, bf16>(ctx, &nf4_lut_bf16),
        synth_precomputed_lut: alloc_buffer_with_data::<B, bf16>(ctx, &synth_lut_bf16),
        awq_precomputed_lut: alloc_buffer_with_data::<B, bf16>(ctx, &awq_lut_bf16),
        nf4_codebook_buf: alloc_buffer_with_data::<B, f16>(ctx, &nf4_codebook_half),
    }
}

// ============================== QMV bench ==============================

struct QmvCell {
    shape: &'static str,
    k: usize,
    n: usize,
    m: usize,
    scalar: Stats,
    awq_lut: Stats,
    nf4: Stats,
    nf4_byte256: Stats,
    /// FLUTE's trick: 256-entry pair LUT replicated D times in threadgroup
    /// memory; lane picks copy via `simd_lane & (D-1)`. Tests whether bank
    /// conflicts on the single-copy `nf4-byte256` path are the cause of its
    /// regression. Same math; only the TG bank distribution differs.
    nf4_dup8: Stats,
    nf4_dup16: Stats,
    nf4_dup32: Stats,
    nf4_grafted: Stats,
    nf4_lut256_graft: Stats,
    nf4_shuf8: Stats,
    nf4_shuf16: Stats,
    nf4_shuf32: Stats,
    nf4_tg: Stats,
    nf4_tg_rep: Stats,
    nf4_tg_vec4: Stats,
    nf4_tg_ilp: Stats,
    /// PERF PROBE: Nf4QmvTg with the post-init `threadgroup_barrier` REMOVED.
    /// Outputs intentionally wrong; bounds barrier kernel-entry cost on M4.
    nf4_tg_nobar: Stats,
    /// Correct variant: 8-copy simdgroup-local TG codebook ordered with
    /// `simdgroup_barrier(mem_threadgroup)`. Bit-equivalent to `Nf4QmvConstant`.
    nf4_tg_sbar: Stats,
    /// Production-flexible sibling of `nf4_tg_sbar`: same simdgroup-local TG
    /// codebook + `simdgroup_barrier` layout, but the 16-entry codebook is
    /// loaded from a `const device half*` buffer at dispatch time (vs the
    /// constant `nf4_codebook[16]`). Deciding cell for whether NF4 has a
    /// production-deployable fast path on M4.
    nf4_tg_sbar_dev: Stats,
    nf4_select: Stats,
    /// AWQ-int4 byte-LUT path with hardcoded constexpr flags (no SPECIALIZE).
    /// Sanity-check sibling to `tmpl_nf4` — should match `awq_lut` within ±2pp.
    tmpl_awq: Stats,
    /// NF4 byte-LUT path with hardcoded constexpr flags. The deciding cell:
    /// compare vs `nf4_lut256_graft` (same math, SPECIALIZE-gated).
    tmpl_nf4: Stats,
    /// NEW: NF4 byte-LUT path where the 256-entry bfloat2 LUT is precomputed
    /// CPU-side and bound as a device buffer. The kernel does NOT include
    /// `nf4_common.h`, so the compiler has no compile-time visibility into
    /// the 16 NF4 codebook values. Deciding cell for "compile-time codebook
    /// visibility = killer?".
    nf4_precomputed: Stats,
    /// ABLATION: same kernel as `nf4_precomputed`, but the device-buffer LUT
    /// is built from the SYNTHETIC dyadic-rational codebook (multiples of 1/8
    /// in [-1, +7/8]). Tests whether NF4's irrational codebook values cause
    /// the runtime gap vs AWQ's integer-valued operands. Outcome A: closes to
    /// awq-lut256 -> mantissa class is the killer. Outcome B: matches
    /// nf4_precomputed -> range or some other effect.
    synth_precomp: Stats,
    /// NEW: same kernel/encode as `nf4_precomputed`, but the device-buffer LUT
    /// is populated from `AWQ_NIBBLE_CODEBOOK` (integers 0..15). Decides
    /// whether the closed-form bypass requires INLINE init code or just
    /// clean values from any source.
    awq_precomp: Stats,
}

fn bench_qmv_cell(
    ctx: &Ctx,
    shape: &'static str,
    input_dim: usize,
    output_dim: usize,
    m: usize,
    qmv_bench: &Nf4QmvBench,
) -> QmvCell {
    println!("=== QMV {} | K={} N={} | M={} ===", shape, input_dim, output_dim, m);
    let bufs = build_buffers(ctx, input_dim, output_dim, m);

    // BASELINE: scalar int4 (QmvFast, use_zero_points=true,
    // use_mlx_quant=false, use_lut=FALSE, no bias).
    let scalar_kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        true,  // use_zero_points
        false, // use_mlx_quant
        false, // use_hadamard
        false, // use_lut: scalar mantissa-trick dequant
        false, // use_nf4
    )
    .expect("scalar QmvFast kernel build");

    // awq-lut256: SAME kernel/inputs, use_lut=TRUE.
    let lut_kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        true,  // use_zero_points
        false, // use_mlx_quant
        false, // use_hadamard
        true,  // use_lut: 256-entry threadgroup LUT
        false, // use_nf4
    )
    .expect("awq-lut256 QmvFast kernel build");

    // nf4-grafted: SAME QmvFast skeleton, ONLY the per-weight dequant swapped
    // to the 16-entry NF4 codebook (scale-only, no zero-points). Built with
    // use_zero_points=FALSE so no zp buffer is needed (graft skips zp load).
    let nf4_graft_kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        false, // use_zero_points: NF4 graft is scale-only
        false, // use_mlx_quant
        false, // use_hadamard
        false, // use_lut
        true,  // use_nf4: 16-entry NF4 codebook dequant in QmvFast skeleton
    )
    .expect("nf4-grafted QmvFast kernel build");

    // nf4-lut256-graft: SAME QmvFast skeleton with use_lut=TRUE AND
    // use_nf4=TRUE. The 256-entry threadgroup half2 LUT is populated at
    // kernel start with byte-batched NF4 codebook pairs; the inner dequant
    // reuses qdot_q4_byte_lut_half (bias=0 via use_zero_points=false).
    // This is the A vs B experiment: does byte-batched LUT mechanism
    // transfer to the NF4 codebook when grafted into the awq-lut256 kernel?
    let nf4_lut_graft_kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        false, // use_zero_points: NF4 graft is scale-only
        false, // use_mlx_quant
        false, // use_hadamard
        true,  // use_lut: 256-entry threadgroup LUT
        true,  // use_nf4: LUT populated with NF4 codebook pairs
    )
    .expect("nf4-lut256-graft QmvFast kernel build");

    let mut y_s = bufs.y_buf.clone();
    let scalar = time_batched(ctx, |enc| {
        scalar_kernel.encode(
            &bufs.w_u32,
            &bufs.s_bf16,
            Some(&bufs.zp_scalar),
            None::<&<B as Backend>::DenseBuffer>,
            &bufs.x_buf,
            &mut y_s,
            None::<&<B as Backend>::DenseBuffer>,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_a = bufs.y_buf.clone();
    let awq_lut = time_batched(ctx, |enc| {
        lut_kernel.encode(
            &bufs.w_u32,
            &bufs.s_bf16,
            Some(&bufs.zp_scalar),
            None::<&<B as Backend>::DenseBuffer>,
            &bufs.x_buf,
            &mut y_a,
            None::<&<B as Backend>::DenseBuffer>,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_n = bufs.y_buf.clone();
    let nf4 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Constant,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_n,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_b = bufs.y_buf.clone();
    let nf4_byte256 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Byte256,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_b,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_d8 = bufs.y_buf.clone();
    let nf4_dup8 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Byte256Dup8,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_d8,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_d16 = bufs.y_buf.clone();
    let nf4_dup16 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Byte256Dup16,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_d16,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_d32 = bufs.y_buf.clone();
    let nf4_dup32 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Byte256Dup32,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_d32,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // nf4-grafted: NF4 codebook dequant inside the QmvFast skeleton. Feed it
    // the SAME NF4 packed u8 weights + SAME bf16 scale as nf4-const (the
    // kernel reads weights as uint8_t* with identical byte/nibble stride), no
    // zero-points (use_zero_points=false → graft skips the zp load).
    let mut y_g = bufs.y_buf.clone();
    let nf4_grafted = time_batched(ctx, |enc| {
        nf4_graft_kernel.encode(
            &bufs.w_u8,
            &bufs.s_bf16,
            None::<&<B as Backend>::DenseBuffer>,
            None::<&<B as Backend>::DenseBuffer>,
            &bufs.x_buf,
            &mut y_g,
            None::<&<B as Backend>::DenseBuffer>,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // nf4-lut256-graft: byte-batched NF4-codebook LUT inside the production
    // QmvFast skeleton. Same SAME NF4 packed u8 weights + bf16 scale, no
    // zero-points.
    let mut y_lg = bufs.y_buf.clone();
    let nf4_lut256_graft = time_batched(ctx, |enc| {
        nf4_lut_graft_kernel.encode(
            &bufs.w_u8,
            &bufs.s_bf16,
            None::<&<B as Backend>::DenseBuffer>,
            None::<&<B as Backend>::DenseBuffer>,
            &bufs.x_buf,
            &mut y_lg,
            None::<&<B as Backend>::DenseBuffer>,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_h8 = bufs.y_buf.clone();
    let nf4_shuf8 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Shuffle8,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_h8,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_h16 = bufs.y_buf.clone();
    let nf4_shuf16 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Shuffle16,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_h16,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_h32 = bufs.y_buf.clone();
    let nf4_shuf32 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Shuffle32,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_h32,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_tg = bufs.y_buf.clone();
    let nf4_tg = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Tg,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_tg,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_tg_rep = bufs.y_buf.clone();
    let nf4_tg_rep = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::TgReplicated,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_tg_rep,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_tg_v4 = bufs.y_buf.clone();
    let nf4_tg_vec4 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::TgVec4,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_tg_v4,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_tg_il = bufs.y_buf.clone();
    let nf4_tg_ilp = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::TgIlp,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_tg_il,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // PERF PROBE: TgNoBarrier (intentionally incorrect outputs — race on TG
    // codebook). Times the same kernel as `Tg` with the post-init
    // threadgroup_barrier deleted, isolating barrier kernel-entry cost.
    let mut y_tg_nb = bufs.y_buf.clone();
    let nf4_tg_nobar = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::TgNoBarrier,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_tg_nb,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // Correct simdgroup-local TG codebook variant ordered with
    // `simdgroup_barrier(mem_threadgroup)` — cheaper than the full
    // threadgroup barrier in `Tg` (only syncs the 32 lanes of this simdgroup).
    let mut y_tg_sb = bufs.y_buf.clone();
    let nf4_tg_sbar = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::TgSimdbar,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_tg_sb,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // Production-flexible simdbar variant: same kernel as `nf4_tg_sbar` but
    // the 16-entry codebook comes from a CPU-built device buffer.
    let mut y_tg_sbd = bufs.y_buf.clone();
    let nf4_tg_sbar_dev = time_batched(ctx, |enc| {
        qmv_bench.encode_tg_simdbar_devbuf(
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.nf4_codebook_buf,
            &bufs.x_buf,
            &mut y_tg_sbd,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_sel = bufs.y_buf.clone();
    let nf4_select = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::Select,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_sel,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // qmv_fast_tmpl_awq: SAME buffer set as awq-lut256 (u32-packed weights,
    // bf16 scale, packed zp). Compile-time constexpr flags (no SPECIALIZE).
    let mut y_ta = bufs.y_buf.clone();
    let tmpl_awq = time_batched(ctx, |enc| {
        qmv_bench.encode_tmpl_awq(
            &bufs.w_u32,
            &bufs.s_bf16,
            &bufs.zp_scalar,
            &bufs.x_buf,
            &mut y_ta,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // qmv_fast_tmpl_nf4: SAME buffer set as nf4-lut-grft (u8 weights, bf16
    // scale, NO zp). Compile-time constexpr flags (no SPECIALIZE).
    let mut y_tn = bufs.y_buf.clone();
    let tmpl_nf4 = time_batched(ctx, |enc| {
        qmv_bench.encode(
            Nf4Variant::QmvFastTemplateNf4Lut,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_tn,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // nf4-precomputed: NEW kernel. Same NF4 weights/scales as nf4-lut-grft,
    // but the 256-entry bfloat2 LUT comes from a device buffer precomputed
    // CPU-side (kernel does NOT include `nf4_common.h` → no compile-time
    // visibility into the codebook values).
    let mut y_np = bufs.y_buf.clone();
    let nf4_precomputed = time_batched(ctx, |enc| {
        qmv_bench.encode_nf4_precomputed(
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.nf4_precomputed_lut,
            &bufs.x_buf,
            &mut y_np,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // synth-precomp: SAME kernel/encode as `nf4-precomp`, only the 256-entry
    // bfloat2 LUT buffer differs (synthetic dyadic-rational codebook). NO
    // kernel binary change — pure ablation of operand mantissa class.
    let mut y_sp = bufs.y_buf.clone();
    let synth_precomp = time_batched(ctx, |enc| {
        qmv_bench.encode_nf4_precomputed(
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.synth_precomputed_lut,
            &bufs.x_buf,
            &mut y_sp,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    // awq-precomp: SAME kernel/encode as `nf4-precomp`, only the 256-entry
    // bfloat2 LUT buffer differs (AWQ nibble codebook = integers 0..15).
    // Decides INLINE-source vs clean-values question.
    let mut y_ap = bufs.y_buf.clone();
    let awq_precomp = time_batched(ctx, |enc| {
        qmv_bench.encode_nf4_precomputed(
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.awq_precomputed_lut,
            &bufs.x_buf,
            &mut y_ap,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    println!("  scalar      : {:.4} ±{:.4} ms", scalar.mean, scalar.std);
    println!("  awq-lut256  : {:.4} ±{:.4} ms  Δ {:+.1}%", awq_lut.mean, awq_lut.std, dpct(&scalar, &awq_lut));
    println!("  nf4-const   : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4.mean, nf4.std, dpct(&scalar, &nf4));
    println!(
        "  nf4-byte256 : {:.4} ±{:.4} ms  Δ {:+.1}%",
        nf4_byte256.mean,
        nf4_byte256.std,
        dpct(&scalar, &nf4_byte256)
    );
    println!("  nf4-dup8    : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_dup8.mean, nf4_dup8.std, dpct(&scalar, &nf4_dup8));
    println!("  nf4-dup16   : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_dup16.mean, nf4_dup16.std, dpct(&scalar, &nf4_dup16));
    println!("  nf4-dup32   : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_dup32.mean, nf4_dup32.std, dpct(&scalar, &nf4_dup32));
    println!(
        "  nf4-grafted : {:.4} ±{:.4} ms  Δ {:+.1}%",
        nf4_grafted.mean,
        nf4_grafted.std,
        dpct(&scalar, &nf4_grafted)
    );
    println!(
        "  nf4-lut-grft: {:.4} ±{:.4} ms  Δ {:+.1}%",
        nf4_lut256_graft.mean,
        nf4_lut256_graft.std,
        dpct(&scalar, &nf4_lut256_graft)
    );
    println!("  nf4-shuf8   : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_shuf8.mean, nf4_shuf8.std, dpct(&scalar, &nf4_shuf8));
    println!("  nf4-shuf16  : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_shuf16.mean, nf4_shuf16.std, dpct(&scalar, &nf4_shuf16));
    println!("  nf4-shuf32  : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_shuf32.mean, nf4_shuf32.std, dpct(&scalar, &nf4_shuf32));
    println!("  nf4-tg      : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_tg.mean, nf4_tg.std, dpct(&scalar, &nf4_tg));
    println!("  nf4-tg-rep  : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_tg_rep.mean, nf4_tg_rep.std, dpct(&scalar, &nf4_tg_rep));
    println!(
        "  nf4-tg-vec4 : {:.4} ±{:.4} ms  Δ {:+.1}%",
        nf4_tg_vec4.mean,
        nf4_tg_vec4.std,
        dpct(&scalar, &nf4_tg_vec4)
    );
    println!("  nf4-tg-ilp  : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_tg_ilp.mean, nf4_tg_ilp.std, dpct(&scalar, &nf4_tg_ilp));
    println!(
        "  nf4-tg-nobar: {:.4} ±{:.4} ms  Δ {:+.1}%  (PROBE: incorrect outputs)",
        nf4_tg_nobar.mean,
        nf4_tg_nobar.std,
        dpct(&scalar, &nf4_tg_nobar)
    );
    println!(
        "  nf4-tg-sbar : {:.4} ±{:.4} ms  Δ {:+.1}%",
        nf4_tg_sbar.mean,
        nf4_tg_sbar.std,
        dpct(&scalar, &nf4_tg_sbar)
    );
    println!(
        "  nf4-tg-sbar-dev: {:.4} ±{:.4} ms  Δ {:+.1}%",
        nf4_tg_sbar_dev.mean,
        nf4_tg_sbar_dev.std,
        dpct(&scalar, &nf4_tg_sbar_dev)
    );
    println!("  nf4-select  : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4_select.mean, nf4_select.std, dpct(&scalar, &nf4_select));
    println!("  tmpl-awq    : {:.4} ±{:.4} ms  Δ {:+.1}%", tmpl_awq.mean, tmpl_awq.std, dpct(&scalar, &tmpl_awq));
    println!("  tmpl-nf4    : {:.4} ±{:.4} ms  Δ {:+.1}%", tmpl_nf4.mean, tmpl_nf4.std, dpct(&scalar, &tmpl_nf4));
    println!(
        "  nf4-precomp : {:.4} ±{:.4} ms  Δ {:+.1}%",
        nf4_precomputed.mean,
        nf4_precomputed.std,
        dpct(&scalar, &nf4_precomputed)
    );
    println!(
        "  synth-prec  : {:.4} ±{:.4} ms  Δ {:+.1}%",
        synth_precomp.mean,
        synth_precomp.std,
        dpct(&scalar, &synth_precomp)
    );
    println!(
        "  awq-precomp : {:.4} ±{:.4} ms  Δ {:+.1}%",
        awq_precomp.mean,
        awq_precomp.std,
        dpct(&scalar, &awq_precomp)
    );

    QmvCell {
        shape,
        k: input_dim,
        n: output_dim,
        m,
        scalar,
        awq_lut,
        nf4,
        nf4_byte256,
        nf4_dup8,
        nf4_dup16,
        nf4_dup32,
        nf4_grafted,
        nf4_lut256_graft,
        nf4_shuf8,
        nf4_shuf16,
        nf4_shuf32,
        nf4_tg,
        nf4_tg_rep,
        nf4_tg_vec4,
        nf4_tg_ilp,
        nf4_tg_nobar,
        nf4_tg_sbar,
        nf4_tg_sbar_dev,
        nf4_select,
        tmpl_awq,
        tmpl_nf4,
        nf4_precomputed,
        synth_precomp,
        awq_precomp,
    }
}

#[test]
#[ignore]
fn qmv_lut_bench() {
    let ctx = Ctx::new().expect("Metal context required");
    let device = {
        use metal::MTLDeviceExt;
        ctx.device.name()
    };
    println!("[QMV_LUT_BENCH] device={}", device);
    println!(
        "[QMV_LUT_BENCH] N_DISPATCH={} WARMUP_BATCHES={} MEASURE_BATCHES={} DROP={}",
        N_DISPATCH, WARMUP_BATCHES, MEASURE_BATCHES, DROP
    );
    println!(
        "[QMV_LUT_BENCH] baseline = scalar int4 (QmvFast use_zero_points=true \
         use_mlx_quant=false use_lut=false, no bias). Challengers: awq-lut256 \
         (same kernel/inputs, use_lut=true), nf4-const (Nf4QmvConstant \
         16-entry codebook, bf16 scale) and nf4-byte256 (Nf4QmvByte256: same \
         NF4 math via byte-batched 256-entry threadgroup half2 LUT). FORMAT \
         CONFOUND: uniform int4 vs NF codebook."
    );

    let qmv_bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    // QMV shapes (label = K x N), M in {1,2,4}.
    let shapes: &[(&'static str, usize, usize)] = &[
        ("LFM-2048", 2048, 2048),
        ("Qwen-MLPup", 896, 4864),
        ("Qwen-MLPdown", 4864, 896),
        ("Llama-4096", 4096, 4096),
        ("Llama-MLPup", 4096, 14336),
        ("Llama-MLPdown", 14336, 4096),
    ];

    let mut results: Vec<QmvCell> = Vec::new();
    for &(shape, k, n) in shapes {
        for &m in &[1usize, 2, 4] {
            results.push(bench_qmv_cell(&ctx, shape, k, n, m, &qmv_bench));
        }
    }

    // Sanity gate: scalar @ LFM-2048 M=1 must be within ~±40% of the known
    // good 0.0248 ms (and not ~0.1 ms, which signals broken batched timing).
    {
        let gate = results.iter().find(|r| r.shape == "LFM-2048" && r.m == 1).expect("LFM-2048 M=1 cell present");
        const GATE_REF_MS: f64 = 0.0248;
        let lo = GATE_REF_MS * 0.50;
        let hi = GATE_REF_MS * 1.60;
        println!(
            "[QMV_LUT_BENCH] sanity gate: scalar LFM-2048 M=1 = {:.4} ms (expect {:.4}±40% => [{:.4}, {:.4}])",
            gate.scalar.mean, GATE_REF_MS, lo, hi
        );
        assert!(
            gate.scalar.mean >= lo && gate.scalar.mean <= hi,
            "SANITY GATE FAILED: scalar LFM-2048 M=1 = {:.4} ms outside [{:.4}, {:.4}] \
             (batched timing likely broken; refusing to report result tables)",
            gate.scalar.mean,
            lo,
            hi
        );
    }

    println!();
    println!("================== QMV LUT bench summary (baseline = scalar) ==================");
    println!(
        "{:<14} {:>11} {:>3} {:>18} {:>14} {:>12} {:>14} {:>12} {:>12} {:>12} {:>14} {:>16} {:>12} {:>12} {:>12} {:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>18} {:>12} {:>14} {:>14} {:>14} {:>15} {:>14}",
        "Shape",
        "KxN",
        "M",
        "scalar ms(±σ)",
        "awq-lut256 Δ%",
        "nf4-const Δ%",
        "nf4-byte256 Δ%",
        "nf4-dup8 Δ%",
        "nf4-dup16 Δ%",
        "nf4-dup32 Δ%",
        "nf4-grafted Δ%",
        "nf4-lut-grft Δ%",
        "nf4-shuf8 Δ%",
        "nf4-shuf16 Δ%",
        "nf4-shuf32 Δ%",
        "nf4-tg Δ%",
        "nf4-tg-rep Δ%",
        "nf4-tg-vec4 Δ%",
        "nf4-tg-ilp Δ%",
        "nf4-tg-nobar Δ%",
        "nf4-tg-sbar Δ%",
        "nf4-tg-sbar-dev Δ%",
        "nf4-select Δ%",
        "tmpl-awq Δ%",
        "tmpl-nf4 Δ%",
        "nf4-precomp Δ%",
        "synth-precomp Δ%",
        "awq-precomp Δ%"
    );
    for r in &results {
        println!(
            "{:<14} {:>11} {:>3} {:>10.4} ±{:<6.4} {:>+13.1}% {:>+11.1}% {:>+13.1}% {:>+11.1}% {:>+11.1}% {:>+11.1}% {:>+13.1}% {:>+15.1}% {:>+11.1}% {:>+11.1}% {:>+11.1}% {:>+11.1}% {:>+13.1}% {:>+13.1}% {:>+13.1}% {:>+13.1}% {:>+13.1}% {:>+17.1}% {:>+11.1}% {:>+13.1}% {:>+13.1}% {:>+13.1}% {:>+14.1}% {:>+13.1}%",
            r.shape,
            format!("{}x{}", r.k, r.n),
            r.m,
            r.scalar.mean,
            r.scalar.std,
            dpct(&r.scalar, &r.awq_lut),
            dpct(&r.scalar, &r.nf4),
            dpct(&r.scalar, &r.nf4_byte256),
            dpct(&r.scalar, &r.nf4_dup8),
            dpct(&r.scalar, &r.nf4_dup16),
            dpct(&r.scalar, &r.nf4_dup32),
            dpct(&r.scalar, &r.nf4_grafted),
            dpct(&r.scalar, &r.nf4_lut256_graft),
            dpct(&r.scalar, &r.nf4_shuf8),
            dpct(&r.scalar, &r.nf4_shuf16),
            dpct(&r.scalar, &r.nf4_shuf32),
            dpct(&r.scalar, &r.nf4_tg),
            dpct(&r.scalar, &r.nf4_tg_rep),
            dpct(&r.scalar, &r.nf4_tg_vec4),
            dpct(&r.scalar, &r.nf4_tg_ilp),
            dpct(&r.scalar, &r.nf4_tg_nobar),
            dpct(&r.scalar, &r.nf4_tg_sbar),
            dpct(&r.scalar, &r.nf4_tg_sbar_dev),
            dpct(&r.scalar, &r.nf4_select),
            dpct(&r.scalar, &r.tmpl_awq),
            dpct(&r.scalar, &r.tmpl_nf4),
            dpct(&r.scalar, &r.nf4_precomputed),
            dpct(&r.scalar, &r.synth_precomp),
            dpct(&r.scalar, &r.awq_precomp)
        );
    }
    println!("==============================================================================");
}

// ============================== QMM bench ==============================

struct QmmCell {
    shape: &'static str,
    m: usize,
    bm_label: &'static str,
    scalar: Stats,
    awq_lut: Stats,
    nf4: Stats,
}

fn bench_qmm_cell(
    ctx: &Ctx,
    shape: &'static str,
    input_dim: usize,
    output_dim: usize,
    m: usize,
    qmm_bench: &Nf4QmmBench,
) -> QmmCell {
    let (bm, bk, bn, wm, wn, tile, bm_label) = pick_tile(m);
    println!("=== QMM {} | K={} N={} | M={} | {} ===", shape, input_dim, output_dim, m, bm_label);
    let bufs = build_buffers(ctx, input_dim, output_dim, m);

    // BASELINE: scalar int4 (QmmTransposed, use_zero_points=true,
    // use_mlx_quant=false, use_lut=FALSE, no bias). aligned_n: N % BN == 0.
    let scalar_kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        bm,
        bk,
        bn,
        wm,
        wn,
        true,  // use_zero_points
        false, // use_mlx_quant
        false, // use_hadamard
        true,  // aligned_n
        false, // use_lut: scalar mantissa-trick dequant
    )
    .expect("scalar QmmTransposed kernel build");

    let lut_kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        bm,
        bk,
        bn,
        wm,
        wn,
        true,  // use_zero_points
        false, // use_mlx_quant
        false, // use_hadamard
        true,  // aligned_n
        true,  // use_lut: 256-entry threadgroup LUT
    )
    .expect("awq-lut256 QmmTransposed kernel build");

    let mut y_s = bufs.y_buf.clone();
    let scalar = time_batched(ctx, |enc| {
        scalar_kernel.encode(
            &bufs.w_u32,
            &bufs.s_bf16,
            Some(&bufs.zp_scalar),
            None::<&<B as Backend>::DenseBuffer>,
            &bufs.x_buf,
            &mut y_s,
            None::<&<B as Backend>::DenseBuffer>,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_a = bufs.y_buf.clone();
    let awq_lut = time_batched(ctx, |enc| {
        lut_kernel.encode(
            &bufs.w_u32,
            &bufs.s_bf16,
            Some(&bufs.zp_scalar),
            None::<&<B as Backend>::DenseBuffer>,
            &bufs.x_buf,
            &mut y_a,
            None::<&<B as Backend>::DenseBuffer>,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    let mut y_n = bufs.y_buf.clone();
    let nf4 = time_batched(ctx, |enc| {
        qmm_bench.encode(
            Nf4Variant::Constant,
            tile,
            &bufs.w_u8,
            &bufs.s_bf16,
            &bufs.x_buf,
            &mut y_n,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            enc,
        );
    });

    println!("  scalar     : {:.4} ±{:.4} ms", scalar.mean, scalar.std);
    println!("  awq-lut256 : {:.4} ±{:.4} ms  Δ {:+.1}%", awq_lut.mean, awq_lut.std, dpct(&scalar, &awq_lut));
    println!("  nf4        : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4.mean, nf4.std, dpct(&scalar, &nf4));

    QmmCell {
        shape,
        m,
        bm_label,
        scalar,
        awq_lut,
        nf4,
    }
}

#[test]
#[ignore]
fn qmm_lut_bench() {
    let ctx = Ctx::new().expect("Metal context required");
    let device = {
        use metal::MTLDeviceExt;
        ctx.device.name()
    };
    println!("[QMM_LUT_BENCH] device={}", device);
    println!(
        "[QMM_LUT_BENCH] N_DISPATCH={} WARMUP_BATCHES={} MEASURE_BATCHES={} DROP={}",
        N_DISPATCH, WARMUP_BATCHES, MEASURE_BATCHES, DROP
    );
    println!(
        "[QMM_LUT_BENCH] baseline = scalar int4 (QmmTransposed use_zero_points=true \
         use_mlx_quant=false use_lut=false, no bias). pick_tile: M<48 -> BM=8, \
         M>=48 -> BM=64, applied identically to all three kernels."
    );

    let qmm_bench = Nf4QmmBench::new(&ctx).expect("Nf4QmmBench build");

    let shapes: &[(&'static str, usize, usize)] = &[("ShapeA-2048x2048", 2048, 2048), ("ShapeB-2560x6912", 2560, 6912)];

    let mut results: Vec<QmmCell> = Vec::new();
    for &m in &[5usize, 8, 16, 32, 48, 64, 128] {
        for &(shape, k, n) in shapes {
            results.push(bench_qmm_cell(&ctx, shape, k, n, m, &qmm_bench));
        }
    }

    println!();
    println!("================== QMM LUT bench summary (baseline = scalar) ==================");
    println!("{:<18} {:>3} {:>5} {:>18} {:>14} {:>12}", "Shape", "M", "BM", "scalar ms(±σ)", "awq-lut256 Δ%", "nf4 Δ%");
    for r in &results {
        println!(
            "{:<18} {:>3} {:>5} {:>10.4} ±{:<6.4} {:>+13.1}% {:>+11.1}%",
            r.shape,
            r.m,
            r.bm_label,
            r.scalar.mean,
            r.scalar.std,
            dpct(&r.scalar, &r.awq_lut),
            dpct(&r.scalar, &r.nf4)
        );
    }
    println!("==============================================================================");
}
