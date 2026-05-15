#![cfg(metal_backend)]

use std::time::Duration;

use backend_uzu::{
    DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{QuantizedMatmulQmmTransposedKernel, QuantizedMatmulQmvFastKernel},
        },
        cpu::nf4_e4m3::f32_to_e4m3,
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

// QMM apples-to-apples rebench: AWQ-int4 (QmmTransposed) vs NF4-const/tg using
// the same BM/BK/BN/WM/WN tile the production AWQ dispatcher would pick.
const WARMUP: usize = 20;
const ITERS: usize = 200;

const GROUP_SIZE: usize = 64;
const BITS: usize = 4;

/// Mirror of the production AWQ dispatcher's tile pick:
///   batch_dim < 48 → small (BM=8/BK=32/BN=32/WM=WN=1)
///   batch_dim ≥ 48 → big   (BM=64/BK=64/BN=64/WM=WN=2)  [bf16 + gs≥64 + N%64==0]
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

#[allow(dead_code)]
struct Stats {
    mean: f64,
    std: f64,
    p25: f64,
    p50: f64,
    p75: f64,
    bins: [usize; 10],
    bin_lo: f64,
    bin_hi: f64,
    bin_width: f64,
}

/// Trimmed mean ± std (drop top/bottom 10%), p50, and a 10-bin histogram over the
/// full sample range.
fn stats(samples_ms: &[f64]) -> Stats {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let trim = n / 10;
    let trimmed = &sorted[trim..n - trim];
    let mean = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
    let var = trimmed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / trimmed.len() as f64;
    let std = var.sqrt();
    let p25 = sorted[n / 4];
    let p50 = sorted[n / 2];
    let p75 = sorted[(3 * n) / 4];

    // Histogram across the full (untrimmed) range.
    let lo = sorted[0];
    let hi = sorted[n - 1];
    let mut bins = [0usize; 10];
    let width = (hi - lo) / 10.0;
    if width > 0.0 {
        for &v in &sorted {
            let mut idx = ((v - lo) / width) as usize;
            if idx >= 10 {
                idx = 9;
            }
            bins[idx] += 1;
        }
    } else {
        bins[0] = n;
    }
    Stats {
        mean,
        std,
        p25,
        p50,
        p75,
        bins,
        bin_lo: lo,
        bin_hi: hi,
        bin_width: width,
    }
}

fn time_loop<F>(mut run_once: F) -> Stats
where
    F: FnMut() -> Duration,
{
    time_loop_n(WARMUP, ITERS, &mut run_once)
}

fn time_loop_n<F>(
    warmup: usize,
    iters: usize,
    run_once: &mut F,
) -> Stats
where
    F: FnMut() -> Duration,
{
    for _ in 0..warmup {
        let _ = run_once();
    }
    let mut samples_ms: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let d = run_once();
        samples_ms.push(d.as_secs_f64() * 1000.0);
    }
    stats(&samples_ms)
}

#[allow(dead_code)]
fn print_histogram(
    label: &str,
    s: &Stats,
) {
    let max_bin = *s.bins.iter().max().unwrap_or(&1);
    let bar_scale = if max_bin > 0 {
        40.0 / max_bin as f64
    } else {
        0.0
    };
    println!("  {} histogram (range {:.4}..{:.4} ms, bin width {:.5} ms):", label, s.bin_lo, s.bin_hi, s.bin_width);
    for (i, &c) in s.bins.iter().enumerate() {
        let lo = s.bin_lo + i as f64 * s.bin_width;
        let hi = lo + s.bin_width;
        let bar_len = (c as f64 * bar_scale) as usize;
        let bar: String = std::iter::repeat('#').take(bar_len).collect();
        println!("    [{:7.4}..{:7.4}] {:>5} {}", lo, hi, c, bar);
    }
}

/// A distribution is bimodal if any non-empty bin in the middle is separated
/// from another non-empty bin by an empty bin, AND there are at least two
/// "peaks" (local maxima) with mass.
fn is_bimodal(bins: &[usize; 10]) -> bool {
    let total: usize = bins.iter().sum();
    if total == 0 {
        return false;
    }
    let threshold = (total as f64 * 0.05) as usize;
    let mut peaks: Vec<usize> = Vec::new();
    for i in 0..bins.len() {
        if bins[i] < threshold {
            continue;
        }
        let left = if i == 0 {
            0
        } else {
            bins[i - 1]
        };
        let right = if i + 1 >= bins.len() {
            0
        } else {
            bins[i + 1]
        };
        if bins[i] >= left && bins[i] >= right {
            peaks.push(i);
        }
    }
    if peaks.len() < 2 {
        return false;
    }
    for w in peaks.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b - a >= 2 {
            let mid_min = (a + 1..b).map(|i| bins[i]).min().unwrap_or(usize::MAX);
            if mid_min < threshold {
                return true;
            }
        }
    }
    false
}

fn modality(bins: &[usize; 10]) -> &'static str {
    if is_bimodal(bins) {
        "bimodal (suspect)"
    } else {
        "unimodal (clean)"
    }
}

#[allow(dead_code)]
struct CellResult {
    shape: &'static str,
    m: usize,
    bm_label: &'static str,
    // Baseline = AWQ-int4 (QmmTransposed).
    awq: Stats,
    cst: Stats,
    tg: Stats,
    e4m3: Stats,
    zp: Stats,
}

/// QMM verdict: lower is better; baseline is AWQ-int4.
fn verdict(delta_pct: f64) -> &'static str {
    if delta_pct.abs() <= 2.0 {
        "tied"
    } else if delta_pct < 0.0 {
        "NF4 wins"
    } else {
        "AWQ wins"
    }
}

/// Build a u8 zero-point buffer packed two 4-bit indices per byte, matching
/// `pack_zero_points` (bits == 4): `zp_stride = (num_groups + 1) / 2` bytes
/// per output row. Realistic random 4-bit indices.
fn build_zp_packed(
    output_dim: usize,
    num_groups: usize,
) -> Vec<u8> {
    let zp_stride = (num_groups + 1) / 2;
    let mut zp_packed = vec![0u8; output_dim * zp_stride];
    for j in 0..output_dim {
        for g in 0..num_groups {
            let idx = ((j * 5 + g * 3 + 1) % 16) as u8;
            let byte = j * zp_stride + g / 2;
            if g % 2 == 0 {
                zp_packed[byte] |= idx & 0x0F;
            } else {
                zp_packed[byte] |= (idx & 0x0F) << 4;
            }
        }
    }
    zp_packed
}

fn bench_cell(
    ctx: &Ctx,
    shape: &'static str,
    input_dim: usize,
    output_dim: usize,
    m: usize,
    qmm_bench: &Nf4QmmBench,
) -> CellResult {
    let (bm, bk, bn, wm, wn, tile, bm_label) = pick_tile(m);
    println!();
    println!(
        "=== {} | K={} N={} | M={} | {} (BM={} BK={} BN={} WM={} WN={}) ===",
        shape, input_dim, output_dim, m, bm_label, bm, bk, bn, wm, wn
    );

    // Static buffers shared across variants.
    let num_groups = input_dim / GROUP_SIZE;
    let packed_bytes_per_row = (input_dim * BITS) / 8;
    let weights_packed: Vec<u8> = (0..(output_dim * packed_bytes_per_row)).map(|i| (i % 251) as u8).collect();
    let scales_f32: Vec<f32> = (0..(output_dim * num_groups)).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
    let biases_f32: Vec<f32> = (0..(output_dim * num_groups)).map(|i| (i % 19) as f32 * 0.125).collect();

    // E4M3: 1-byte FP8 scale, built by round-tripping the SAME f32 scales
    // through E4M3 (fair representation, not random bytes).
    let scale_bytes_e4m3: Vec<u8> = scales_f32.iter().map(|&s| f32_to_e4m3(s)).collect();
    // Zp: realistic random 4-bit zero-point indices, packed two-per-byte.
    let zp_packed = build_zp_packed(output_dim, num_groups);

    let w_buf = alloc_buffer_with_data::<B, u8>(ctx, &weights_packed);
    let s_buf = bf16_buf(ctx, &scales_f32);
    let b_buf = bf16_buf(ctx, &biases_f32);
    let s_e4m3_buf = alloc_buffer_with_data::<B, u8>(ctx, &scale_bytes_e4m3);
    let zp_buf = alloc_buffer_with_data::<B, u8>(ctx, &zp_packed);

    let x_f32: Vec<f32> = (0..(m * input_dim)).map(|i| ((i % 257) as f32) / 257.0).collect();
    let x_buf = bf16_buf(ctx, &x_f32);
    let mut y_buf = ctx.create_buffer(m * output_dim * DataType::BF16.size_in_bytes()).expect("y buf");

    // AWQ-int4 (MLX-bias) via QmmTransposed with the picked tile — matches
    // exactly what the production AWQ dispatcher would emit for this M.
    let awq_kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        bm,
        bk,
        bn,
        wm,
        wn,
        false, // use_zero_points
        true,  // use_mlx_quant (AWQ-style: biases, no zero-points)
        false, // use_hadamard
        true,  // aligned_n: both shapes have N divisible by BN
    )
    .expect("AWQ QmmTransposed kernel build");

    let awq_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        awq_kernel.encode(
            &w_buf,
            &s_buf,
            None::<&<B as Backend>::DenseBuffer>,
            Some(&b_buf),
            &x_buf,
            &mut y_buf,
            None::<&<B as Backend>::DenseBuffer>,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let awq = time_loop(awq_run);

    let nf4_const_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        qmm_bench.encode(
            Nf4Variant::Constant,
            tile,
            &w_buf,
            &s_buf,
            &x_buf,
            &mut y_buf,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let cst = time_loop(nf4_const_run);

    let nf4_tg_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        qmm_bench.encode(
            Nf4Variant::Tg,
            tile,
            &w_buf,
            &s_buf,
            &x_buf,
            &mut y_buf,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let tg = time_loop(nf4_tg_run);

    let nf4_e4m3_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        qmm_bench.encode(
            Nf4Variant::E4m3,
            tile,
            &w_buf,
            &s_e4m3_buf,
            &x_buf,
            &mut y_buf,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let e4m3 = time_loop(nf4_e4m3_run);

    let nf4_zp_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        qmm_bench.encode_zp(
            tile,
            &w_buf,
            &s_buf,
            &zp_buf,
            &x_buf,
            &mut y_buf,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let zp = time_loop(nf4_zp_run);

    let dpct = |s: &Stats| (s.p50 - awq.p50) / awq.p50 * 100.0;
    let delta_const = dpct(&cst);
    let delta_tg = dpct(&tg);
    let delta_e4m3 = dpct(&e4m3);
    let delta_zp = dpct(&zp);

    println!(
        "  AWQ-int4 (QmmTransposed/{}): mean {:.4} ±{:.4} ms  p50 {:.4} ms  [{}]",
        bm_label,
        awq.mean,
        awq.std,
        awq.p50,
        modality(&awq.bins)
    );
    println!(
        "  NF4-const : mean {:.4} ±{:.4} ms  p50 {:.4} ms  [{}]  Δp50 vs AWQ = {:+.2}% ({})",
        cst.mean,
        cst.std,
        cst.p50,
        modality(&cst.bins),
        delta_const,
        verdict(delta_const)
    );
    println!(
        "  NF4-tg    : mean {:.4} ±{:.4} ms  p50 {:.4} ms  [{}]  Δp50 vs AWQ = {:+.2}% ({})",
        tg.mean,
        tg.std,
        tg.p50,
        modality(&tg.bins),
        delta_tg,
        verdict(delta_tg)
    );
    println!(
        "  NF4-e4m3  : mean {:.4} ±{:.4} ms  p50 {:.4} ms  [{}]  Δp50 vs AWQ = {:+.2}% ({})",
        e4m3.mean,
        e4m3.std,
        e4m3.p50,
        modality(&e4m3.bins),
        delta_e4m3,
        verdict(delta_e4m3)
    );
    println!(
        "  NF4-zp    : mean {:.4} ±{:.4} ms  p50 {:.4} ms  [{}]  Δp50 vs AWQ = {:+.2}% ({})",
        zp.mean,
        zp.std,
        zp.p50,
        modality(&zp.bins),
        delta_zp,
        verdict(delta_zp)
    );

    CellResult {
        shape,
        m,
        bm_label,
        awq,
        cst,
        tg,
        e4m3,
        zp,
    }
}

#[test]
#[ignore]
fn qmm_lut_bench() {
    let ctx = Ctx::new().expect("Metal context required");
    println!("[NF4_BENCH] device={}", {
        use metal::MTLDeviceExt;
        ctx.device.name()
    });
    println!("[NF4_BENCH] WARMUP={} ITERS={} (trimmed mean ±σ over middle 80%, p50 over full sample)", WARMUP, ITERS);

    let qmm_bench = Nf4QmmBench::new(&ctx).expect("Nf4QmmBench build");

    let mut results: Vec<CellResult> = Vec::new();
    // Prefill-shaped M sweep × two shapes. M ∈ {5, 16, 32, 64, 128}.
    // pick_tile: M<48 -> Small (BM=8); M>=48 -> Big (BM=64). Same tile is
    // applied to every NF4 variant (apples-to-apples vs AWQ).
    for &m in &[5usize, 16, 32, 64, 128] {
        results.push(bench_cell(&ctx, "ShapeA-2048x2048", 2048, 2048, m, &qmm_bench));
        results.push(bench_cell(&ctx, "ShapeB-2560x6912", 2560, 6912, m, &qmm_bench));
    }

    // Final combined table. Baseline for Δ = AWQ-int4 (QmmTransposed).
    println!();
    println!("================== QMM LUT bench summary (baseline = AWQ-int4) ==================");
    println!(
        "{:<18} {:>4} {:>5} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8} {:>8} {:>8}",
        "Shape", "M", "BM", "AWQ p50", "NF4c p50", "NF4tg p50", "E4M3 p50", "Zp p50", "Δ c", "Δ tg", "Δ e4m3", "Δ zp"
    );
    for r in &results {
        let d = |s: &Stats| (s.p50 - r.awq.p50) / r.awq.p50 * 100.0;
        println!(
            "{:<18} {:>4} {:>5} {:>9.4}ms {:>9.4}ms {:>9.4}ms {:>9.4}ms {:>9.4}ms {:>+7.2}% {:>+7.2}% {:>+7.2}% {:>+7.2}%",
            r.shape,
            r.m,
            r.bm_label,
            r.awq.p50,
            r.cst.p50,
            r.tg.p50,
            r.e4m3.p50,
            r.zp.p50,
            d(&r.cst),
            d(&r.tg),
            d(&r.e4m3),
            d(&r.zp)
        );
    }
    println!("  (trimmed-mean ±σ + modality flag per cell printed in the detail blocks above)");
    println!("=================================================================================");
}

// ===================== QMV LUT bench (M ∈ {1,2,3,4}) =====================
//
// Decode-regime comparison at small batch (true QMV regime). Baseline for Δ
// is the production `QmvFast` kernel (the main decode QMV).
//
//   * QmvFast  (BASELINE): production `QuantizedMatmulQmvFastKernel`,
//     instantiated MLX-bias form (use_mlx_quant=true, no zero points, no
//     hadamard, gs=64, bf16 act) — i.e. this *is* the AWQ-int4 QMV path the
//     production AWQ dispatcher emits for batch_dim < 5. There is no separate
//     "AWQ QMV" kernel distinct from QmvFast, so one column covers both.
//     FORMAT CONFOUND: QmvFast consumes int4 weights + a bf16 per-group scale
//     + bf16 MLX bias (asymmetric int4). The NF4 variants consume the fixed
//     16-entry NF4 codebook with a symmetric gs=64 scale (E4M3 swaps the
//     scale dtype; Zp adds a 4-bit zero-point LUT). They are NOT the same
//     numeric format — this is a kernel-cost comparison at matched shape, not
//     an isoaccuracy comparison.
//   * NF4-const: `Nf4QmvConstantMetalKernel` (codebook in `constant` space).
//   * NF4-tg:    `Nf4QmvTgMetalKernel`       (codebook in threadgroup mem).
//   * NF4-e4m3:  `Nf4QmvE4m3MetalKernel`     (1-byte FP8 per-group scale).
//   * NF4-zp:    `Nf4QmvZpMetalKernel`       (+4-bit per-group zero-point LUT).
//
// Heavier sampling (50 warmup + 1000 timed) so we can compute p25/p50/p75 and
// a 10-bin histogram per cell. QMV dispatches are tiny so this stays cheap.
const QMV_WARMUP: usize = 50;
const QMV_ITERS: usize = 1000;

#[allow(dead_code)]
struct QmvCellResult {
    shape: &'static str,
    input_dim: usize,
    output_dim: usize,
    m: usize,
    awq: Stats, // QmvFast (baseline)
    cst: Stats,
    tg: Stats,
    e4m3: Stats,
    zp: Stats,
}

fn bimodal_flag(s: &Stats) -> bool {
    // Bimodal flag per spec: p25/p75 spread > 30% of p50.
    if s.p50 <= 0.0 {
        return false;
    }
    (s.p75 - s.p25) / s.p50 > 0.30
}

fn print_hist_compact(
    label: &str,
    s: &Stats,
) {
    print!("    {} hist [{:.4}..{:.4}ms]:", label, s.bin_lo, s.bin_hi);
    for c in &s.bins {
        print!(" {}", c);
    }
    println!();
}

fn bench_qmv_cell(
    ctx: &Ctx,
    shape: &'static str,
    input_dim: usize,
    output_dim: usize,
    m: usize,
    qmv_bench: &Nf4QmvBench,
) -> QmvCellResult {
    println!();
    println!("=== QMV  {} | K={} N={} | M={} ===", shape, input_dim, output_dim, m);

    let num_groups = input_dim / GROUP_SIZE;
    let packed_bytes_per_row = (input_dim * BITS) / 8;
    let weights_packed: Vec<u8> = (0..(output_dim * packed_bytes_per_row)).map(|i| (i % 251) as u8).collect();
    let scales_f32: Vec<f32> = (0..(output_dim * num_groups)).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
    let biases_f32: Vec<f32> = (0..(output_dim * num_groups)).map(|i| (i % 19) as f32 * 0.125).collect();

    let scale_bytes_e4m3: Vec<u8> = scales_f32.iter().map(|&s| f32_to_e4m3(s)).collect();
    let zp_packed = build_zp_packed(output_dim, num_groups);

    let w_buf = alloc_buffer_with_data::<B, u8>(ctx, &weights_packed);
    let s_buf = bf16_buf(ctx, &scales_f32);
    let b_buf = bf16_buf(ctx, &biases_f32);
    let s_e4m3_buf = alloc_buffer_with_data::<B, u8>(ctx, &scale_bytes_e4m3);
    let zp_buf = alloc_buffer_with_data::<B, u8>(ctx, &zp_packed);

    let x_f32: Vec<f32> = (0..(m * input_dim)).map(|i| ((i % 257) as f32) / 257.0).collect();
    let x_buf = bf16_buf(ctx, &x_f32);
    let mut y_buf = ctx.create_buffer(m * output_dim * DataType::BF16.size_in_bytes()).expect("y buf");

    // QmvFast (baseline): production decode QMV, MLX bias form (= AWQ-int4 QMV).
    let awq_kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        false, // use_zero_points
        true,  // use_mlx_quant
        false, // use_hadamard
    )
    .expect("AWQ QmvFast kernel build");

    let mut awq_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        awq_kernel.encode(
            &w_buf,
            &s_buf,
            None::<&<B as Backend>::DenseBuffer>,
            Some(&b_buf),
            &x_buf,
            &mut y_buf,
            None::<&<B as Backend>::DenseBuffer>,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let awq = time_loop_n(QMV_WARMUP, QMV_ITERS, &mut awq_run);

    let mut cst_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        qmv_bench.encode(
            Nf4Variant::Constant,
            &w_buf,
            &s_buf,
            &x_buf,
            &mut y_buf,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let cst = time_loop_n(QMV_WARMUP, QMV_ITERS, &mut cst_run);

    let mut tg_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        qmv_bench.encode(
            Nf4Variant::Tg,
            &w_buf,
            &s_buf,
            &x_buf,
            &mut y_buf,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let tg = time_loop_n(QMV_WARMUP, QMV_ITERS, &mut tg_run);

    let mut e4m3_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        qmv_bench.encode(
            Nf4Variant::E4m3,
            &w_buf,
            &s_e4m3_buf,
            &x_buf,
            &mut y_buf,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let e4m3 = time_loop_n(QMV_WARMUP, QMV_ITERS, &mut e4m3_run);

    let mut zp_run = || -> Duration {
        let mut encoder = Encoder::new(ctx).unwrap();
        qmv_bench.encode_zp(
            &w_buf,
            &s_buf,
            &zp_buf,
            &x_buf,
            &mut y_buf,
            input_dim as u32,
            output_dim as u32,
            m as u32,
            &mut encoder,
        );
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        completed.gpu_execution_time()
    };
    let zp = time_loop_n(QMV_WARMUP, QMV_ITERS, &mut zp_run);

    let dpct = |s: &Stats| (s.p50 - awq.p50) / awq.p50 * 100.0;
    let dc = dpct(&cst);
    let dt = dpct(&tg);
    let de = dpct(&e4m3);
    let dz = dpct(&zp);

    println!(
        "  QmvFast (baseline): mean {:.4} ±{:.4} ms  p25 {:.4} p50 {:.4} p75 {:.4}  [{}]",
        awq.mean,
        awq.std,
        awq.p25,
        awq.p50,
        awq.p75,
        modality(&awq.bins)
    );
    print_hist_compact("Qmv", &awq);
    println!(
        "  NF4-const  : mean {:.4} ±{:.4} ms  p25 {:.4} p50 {:.4} p75 {:.4}  [{}]  Δp50 {:+.2}% ({})",
        cst.mean,
        cst.std,
        cst.p25,
        cst.p50,
        cst.p75,
        modality(&cst.bins),
        dc,
        verdict(dc)
    );
    print_hist_compact("NF4c", &cst);
    println!(
        "  NF4-tg     : mean {:.4} ±{:.4} ms  p25 {:.4} p50 {:.4} p75 {:.4}  [{}]  Δp50 {:+.2}% ({})",
        tg.mean,
        tg.std,
        tg.p25,
        tg.p50,
        tg.p75,
        modality(&tg.bins),
        dt,
        verdict(dt)
    );
    print_hist_compact("NF4tg", &tg);
    println!(
        "  NF4-e4m3   : mean {:.4} ±{:.4} ms  p25 {:.4} p50 {:.4} p75 {:.4}  [{}]  Δp50 {:+.2}% ({})",
        e4m3.mean,
        e4m3.std,
        e4m3.p25,
        e4m3.p50,
        e4m3.p75,
        modality(&e4m3.bins),
        de,
        verdict(de)
    );
    print_hist_compact("NF4e4m3", &e4m3);
    println!(
        "  NF4-zp     : mean {:.4} ±{:.4} ms  p25 {:.4} p50 {:.4} p75 {:.4}  [{}]  Δp50 {:+.2}% ({})",
        zp.mean,
        zp.std,
        zp.p25,
        zp.p50,
        zp.p75,
        modality(&zp.bins),
        dz,
        verdict(dz)
    );
    print_hist_compact("NF4zp", &zp);

    QmvCellResult {
        shape,
        input_dim,
        output_dim,
        m,
        awq,
        cst,
        tg,
        e4m3,
        zp,
    }
}

#[test]
#[ignore]
fn qmv_lut_bench() {
    let ctx = Ctx::new().expect("Metal context required");
    println!("[QMV_BENCH] device={}", {
        use metal::MTLDeviceExt;
        ctx.device.name()
    });
    println!(
        "[QMV_BENCH] WARMUP={} ITERS={} (trimmed mean over middle 80%; p25/p50/p75 over full sample)",
        QMV_WARMUP, QMV_ITERS
    );

    let qmv_bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");
    println!(
        "[QMV_BENCH] baseline = QmvFast (production decode QMV; == AWQ-int4 QMV path). \
         FORMAT CONFOUND: QmvFast = asymmetric int4 + bf16 scale + MLX bias; \
         NF4 = fixed 16-entry codebook + symmetric gs=64 scale (E4M3 = FP8 \
         scale dtype; Zp = +4-bit zero-point LUT). Matched-shape kernel-cost \
         comparison, NOT isoaccuracy."
    );

    // The two real shapes the harness already uses (decode-shaped, M small).
    let shapes: &[(&'static str, usize, usize)] = &[("ShapeA-2048x2048", 2048, 2048), ("ShapeB-2560x6912", 2560, 6912)];

    let mut results: Vec<QmvCellResult> = Vec::new();
    for &(shape, k, n) in shapes {
        for &m in &[1usize, 2, 3, 4] {
            results.push(bench_qmv_cell(&ctx, shape, k, n, m, &qmv_bench));
        }
    }

    // Summary tables: one per M, shape × variant. Baseline = QmvFast.
    println!();
    println!("================== QMV LUT bench summary (baseline = QmvFast) ==================");
    for &m_target in &[1usize, 2, 3, 4] {
        println!();
        println!("--- M = {} ---", m_target);
        println!(
            "{:<18} {:>11} {:>11} {:>11} {:>11} {:>11} {:>7} {:>7} {:>7} {:>7}  flags",
            "Shape", "Qmv p50", "NF4c p50", "NF4tg p50", "E4M3 p50", "Zp p50", "Δ c", "Δ tg", "Δ e4m3", "Δ zp"
        );
        for r in results.iter().filter(|r| r.m == m_target) {
            let d = |s: &Stats| (s.p50 - r.awq.p50) / r.awq.p50 * 100.0;
            let (dc, dt, de, dz) = (d(&r.cst), d(&r.tg), d(&r.e4m3), d(&r.zp));
            let mut flags: Vec<&str> = Vec::new();
            if bimodal_flag(&r.awq) {
                flags.push("Qmv-bimodal");
            }
            if bimodal_flag(&r.cst) {
                flags.push("NF4c-bimodal");
            }
            if bimodal_flag(&r.tg) {
                flags.push("NF4tg-bimodal");
            }
            if bimodal_flag(&r.e4m3) {
                flags.push("E4M3-bimodal");
            }
            if bimodal_flag(&r.zp) {
                flags.push("Zp-bimodal");
            }
            println!(
                "{:<18} {:>8.4}ms {:>8.4}ms {:>8.4}ms {:>8.4}ms {:>8.4}ms {:>+6.2}% {:>+6.2}% {:>+6.2}% {:>+6.2}%  {}",
                r.shape,
                r.awq.p50,
                r.cst.p50,
                r.tg.p50,
                r.e4m3.p50,
                r.zp.p50,
                dc,
                dt,
                de,
                dz,
                flags.join(",")
            );
        }
    }
    println!("  σ (trimmed mean): printed per cell in the detail blocks above.");
    println!("===============================================================================");

    // M-scaling per shape per variant: ratio p50_M / p50_{M-1}.
    println!();
    println!("--- NF4 M-scaling ratios (p50_M / p50_{{M-1}}) ---");
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Shape", "c 2/1", "c 3/2", "c 4/3", "tg 2/1", "e4m3 2/1", "zp 2/1"
    );
    for &(shape, _k, _n) in shapes {
        let by_m: std::collections::HashMap<usize, &QmvCellResult> =
            results.iter().filter(|r| r.shape == shape).map(|r| (r.m, r)).collect();
        let (r1, r2, r3, r4) = (by_m[&1], by_m[&2], by_m[&3], by_m[&4]);
        println!(
            "{:<18} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            shape,
            r2.cst.p50 / r1.cst.p50,
            r3.cst.p50 / r2.cst.p50,
            r4.cst.p50 / r3.cst.p50,
            r2.tg.p50 / r1.tg.p50,
            r2.e4m3.p50 / r1.e4m3.p50,
            r2.zp.p50 / r1.zp.p50
        );
    }
    println!();
}
