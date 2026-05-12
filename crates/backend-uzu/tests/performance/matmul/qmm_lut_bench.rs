#![cfg(metal_backend)]

use std::time::Duration;

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
    awq_p50: f64,
    awq_mean: f64,
    awq_std: f64,
    cst_p50: f64,
    cst_mean: f64,
    cst_std: f64,
    tg_p50: f64,
    tg_mean: f64,
    tg_std: f64,
}

fn verdict(delta_pct: f64) -> &'static str {
    if delta_pct.abs() <= 2.0 {
        "tied"
    } else if delta_pct < 0.0 {
        "NF4 wins"
    } else {
        "AWQ wins"
    }
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

    let w_buf = alloc_buffer_with_data::<B, u8>(ctx, &weights_packed);
    let s_buf = bf16_buf(ctx, &scales_f32);
    let b_buf = bf16_buf(ctx, &biases_f32);

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

    let delta_const = (cst.p50 - awq.p50) / awq.p50 * 100.0;
    let delta_tg = (tg.p50 - awq.p50) / awq.p50 * 100.0;

    println!(
        "  AWQ-int4 (QmmTransposed/{}): mean {:.4} ±{:.4} ms  p50 {:.4} ms  [{}]",
        bm_label,
        awq.mean,
        awq.std,
        awq.p50,
        modality(&awq.bins)
    );
    println!(
        "  NF4-const         : mean {:.4} ±{:.4} ms  p50 {:.4} ms  [{}]  Δp50 vs AWQ = {:+.2}% ({})",
        cst.mean,
        cst.std,
        cst.p50,
        modality(&cst.bins),
        delta_const,
        verdict(delta_const)
    );
    println!(
        "  NF4-tg            : mean {:.4} ±{:.4} ms  p50 {:.4} ms  [{}]  Δp50 vs AWQ = {:+.2}% ({})",
        tg.mean,
        tg.std,
        tg.p50,
        modality(&tg.bins),
        delta_tg,
        verdict(delta_tg)
    );

    CellResult {
        shape,
        m,
        bm_label,
        awq_p50: awq.p50,
        awq_mean: awq.mean,
        awq_std: awq.std,
        cst_p50: cst.p50,
        cst_mean: cst.mean,
        cst_std: cst.std,
        tg_p50: tg.p50,
        tg_mean: tg.mean,
        tg_std: tg.std,
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
    // Full M sweep × two shapes. M ∈ {5, 8, 16, 32, 48, 64, 128}.
    for &m in &[5usize, 8, 16, 32, 48, 64, 128] {
        results.push(bench_cell(&ctx, "ShapeA-2048x2048", 2048, 2048, m, &qmm_bench));
        results.push(bench_cell(&ctx, "ShapeB-2560x6912", 2560, 6912, m, &qmm_bench));
    }

    // Final combined table.
    println!();
    println!("================== QMM LUT bench summary ==================");
    println!(
        "{:<18} {:>4} {:>6} {:>14} {:>14} {:>14} {:>10} {:>10} {:>12} {:>12}",
        "Shape", "M", "BM", "AWQ-int4 p50", "NF4-const p50", "NF4-tg p50", "Δ const", "Δ tg", "const", "tg"
    );
    for r in &results {
        let dc = (r.cst_p50 - r.awq_p50) / r.awq_p50 * 100.0;
        let dt = (r.tg_p50 - r.awq_p50) / r.awq_p50 * 100.0;
        println!(
            "{:<18} {:>4} {:>6} {:>11.4} ms {:>11.4} ms {:>11.4} ms {:>+9.2}% {:>+9.2}% {:>12} {:>12}",
            r.shape,
            r.m,
            r.bm_label,
            r.awq_p50,
            r.cst_p50,
            r.tg_p50,
            dc,
            dt,
            verdict(dc),
            verdict(dt)
        );
    }
    println!("===========================================================");
}

// ===================== QMV LUT bench (M ∈ {1,2,4}) =====================
//
// Three-way comparison at small batch (true QMV regime):
//   * AWQ-int4: production `QuantizedMatmulQmvFastKernel` (MLX bias form, gs=64,
//     bf16 act, no hadamard, no zero points) — exactly what the production
//     AWQ dispatcher would emit when batch_dim < 5 (it always falls to QmvFast
//     in that range).
//   * NF4-const: `Nf4QmvConstantMetalKernel` (codebook in `constant` space).
//   * NF4-tg:    `Nf4QmvTgMetalKernel`       (codebook cooperatively loaded
//                into threadgroup memory).
//
// Heavier sampling (50 warmup + 1000 timed) so we can compute p25/p50/p75 and
// a 10-bin histogram per cell.
const QMV_WARMUP: usize = 50;
const QMV_ITERS: usize = 1000;

#[allow(dead_code)]
struct QmvCellResult {
    shape: &'static str,
    input_dim: usize,
    output_dim: usize,
    m: usize,
    awq: Stats,
    cst: Stats,
    tg: Stats,
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

    let w_buf = alloc_buffer_with_data::<B, u8>(ctx, &weights_packed);
    let s_buf = bf16_buf(ctx, &scales_f32);
    let b_buf = bf16_buf(ctx, &biases_f32);

    let x_f32: Vec<f32> = (0..(m * input_dim)).map(|i| ((i % 257) as f32) / 257.0).collect();
    let x_buf = bf16_buf(ctx, &x_f32);
    let mut y_buf = ctx.create_buffer(m * output_dim * DataType::BF16.size_in_bytes()).expect("y buf");

    // AWQ-int4 via production QmvFast (MLX bias form, no zero points, no hadamard).
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

    let dc = (cst.p50 - awq.p50) / awq.p50 * 100.0;
    let dt = (tg.p50 - awq.p50) / awq.p50 * 100.0;

    println!(
        "  AWQ-int4 QmvFast: mean {:.4} ±{:.4} ms  p25 {:.4} p50 {:.4} p75 {:.4}  [{}]",
        awq.mean,
        awq.std,
        awq.p25,
        awq.p50,
        awq.p75,
        modality(&awq.bins)
    );
    print_hist_compact("AWQ", &awq);
    println!(
        "  NF4-const       : mean {:.4} ±{:.4} ms  p25 {:.4} p50 {:.4} p75 {:.4}  [{}]  Δp50 {:+.2}% ({})",
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
        "  NF4-tg          : mean {:.4} ±{:.4} ms  p25 {:.4} p50 {:.4} p75 {:.4}  [{}]  Δp50 {:+.2}% ({})",
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

    QmvCellResult {
        shape,
        input_dim,
        output_dim,
        m,
        awq,
        cst,
        tg,
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

    // 6 production-like shapes × M ∈ {1,2,4} = 18 cells × 3 variants.
    let shapes: &[(&'static str, usize, usize)] = &[
        ("LFM-2048", 2048, 2048),
        ("Qwen-MLPup", 2560, 6912),
        ("Qwen-MLPdown", 6912, 2560),
        ("Llama-4096", 4096, 4096),
        ("Llama-MLPup", 4096, 14336),
        ("Llama-MLPdown", 14336, 4096),
    ];

    let mut results: Vec<QmvCellResult> = Vec::new();
    for &(shape, k, n) in shapes {
        for &m in &[1usize, 2, 4] {
            results.push(bench_qmv_cell(&ctx, shape, k, n, m, &qmv_bench));
        }
    }

    // Summary tables: one per M, shape × variant.
    println!();
    println!("================== QMV LUT bench summary ==================");
    for &m_target in &[1usize, 2, 4] {
        println!();
        println!("--- M = {} ---", m_target);
        println!(
            "{:<16} {:>14} {:>14} {:>14} {:>9} {:>9}  flags",
            "Shape", "AWQ p50 (ms)", "NF4-c p50 (ms)", "NF4-tg p50 (ms)", "Δ const", "Δ tg"
        );
        for r in results.iter().filter(|r| r.m == m_target) {
            let dc = (r.cst.p50 - r.awq.p50) / r.awq.p50 * 100.0;
            let dt = (r.tg.p50 - r.awq.p50) / r.awq.p50 * 100.0;
            let mut flags: Vec<&str> = Vec::new();
            if bimodal_flag(&r.awq) {
                flags.push("AWQ-bimodal");
            }
            if bimodal_flag(&r.cst) {
                flags.push("NF4c-bimodal");
            }
            if bimodal_flag(&r.tg) {
                flags.push("NF4tg-bimodal");
            }
            if dc.abs() < 2.0 {
                flags.push("const-tied");
            }
            if dt.abs() < 2.0 {
                flags.push("tg-tied");
            }
            println!(
                "{:<16} {:>11.4} ms {:>11.4} ms {:>11.4} ms {:>+8.2}% {:>+8.2}%  {}",
                r.shape,
                r.awq.p50,
                r.cst.p50,
                r.tg.p50,
                dc,
                dt,
                flags.join(",")
            );
        }
    }
    println!("===========================================================");

    // M-scaling per shape per variant: ratio M=2/M=1 and M=4/M=2.
    println!();
    println!("--- NF4 M-scaling ratios (p50_M / p50_{{M/2}}) ---");
    println!("{:<16} {:>14} {:>14} {:>14} {:>14}", "Shape", "const 2/1", "const 4/2", "tg 2/1", "tg 4/2");
    for &(shape, _k, _n) in shapes {
        let by_m: std::collections::HashMap<usize, &QmvCellResult> =
            results.iter().filter(|r| r.shape == shape).map(|r| (r.m, r)).collect();
        let r1 = by_m[&1];
        let r2 = by_m[&2];
        let r4 = by_m[&4];
        let c21 = r2.cst.p50 / r1.cst.p50;
        let c42 = r4.cst.p50 / r2.cst.p50;
        let t21 = r2.tg.p50 / r1.tg.p50;
        let t42 = r4.tg.p50 / r2.tg.p50;
        println!("{:<16} {:>14.3} {:>14.3} {:>14.3} {:>14.3}", shape, c21, c42, t21, t42);
    }
    println!();
}
