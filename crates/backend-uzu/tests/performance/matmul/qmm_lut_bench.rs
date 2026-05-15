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
use half::bf16;

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

    CellBuffers {
        w_u32: alloc_buffer_with_data::<B, u32>(ctx, &w_u32_data),
        s_bf16: bf16_buf(ctx, &scales_f32),
        zp_scalar: alloc_buffer_with_data::<B, u8>(ctx, &zp_scalar_packed),
        w_u8: alloc_buffer_with_data::<B, u8>(ctx, &weights_u8),
        x_buf: bf16_buf(ctx, &x_f32),
        y_buf: ctx.create_buffer(m * output_dim * DataType::BF16.size_in_bytes()).expect("y buf"),
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
    )
    .expect("awq-lut256 QmvFast kernel build");

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

    println!("  scalar      : {:.4} ±{:.4} ms", scalar.mean, scalar.std);
    println!("  awq-lut256  : {:.4} ±{:.4} ms  Δ {:+.1}%", awq_lut.mean, awq_lut.std, dpct(&scalar, &awq_lut));
    println!("  nf4-const   : {:.4} ±{:.4} ms  Δ {:+.1}%", nf4.mean, nf4.std, dpct(&scalar, &nf4));
    println!(
        "  nf4-byte256 : {:.4} ±{:.4} ms  Δ {:+.1}%",
        nf4_byte256.mean,
        nf4_byte256.std,
        dpct(&scalar, &nf4_byte256)
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
        let lo = GATE_REF_MS * 0.60;
        let hi = GATE_REF_MS * 1.40;
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
        "{:<14} {:>11} {:>3} {:>18} {:>14} {:>12} {:>14}",
        "Shape", "KxN", "M", "scalar ms(±σ)", "awq-lut256 Δ%", "nf4-const Δ%", "nf4-byte256 Δ%"
    );
    for r in &results {
        println!(
            "{:<14} {:>11} {:>3} {:>10.4} ±{:<6.4} {:>+13.1}% {:>+11.1}% {:>+13.1}%",
            r.shape,
            format!("{}x{}", r.k, r.n),
            r.m,
            r.scalar.mean,
            r.scalar.std,
            dpct(&r.scalar, &r.awq_lut),
            dpct(&r.scalar, &r.nf4),
            dpct(&r.scalar, &r.nf4_byte256)
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
