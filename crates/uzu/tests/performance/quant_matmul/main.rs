#![cfg(metal_backend)]

use indicatif::{ProgressBar, ProgressStyle};
use metal::MTLDeviceExt;
use uzu::{
    DataType,
    backends::{
        common::{Backend, Context, gpu_types::QuantizationMode, kernel::quant_matmul::ForceKernel},
        metal::Metal,
    },
};

use crate::quant_matmul::{
    bench,
    output::{print_comparison_table, print_results_table},
    shapes::TestShape,
};

type Ctx = <Metal as Backend>::Context;

/// Compare QmvFast and QmmTransposedSmall for tokens 1–32 on a single representative shape.
#[test]
#[ignore]
fn quant_matmul_perf() {
    let context = Ctx::new().expect("Metal context required");

    let data_type = DataType::BF16;
    let bits: usize = 4;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;

    let input_dim = 4096;
    let output_dim = 14336;
    let batches: Vec<usize> = (1..=32).collect();

    let kernels = [("QmvFast", ForceKernel::QmvFast), ("QmmSmall", ForceKernel::QmmTransposedSmall)];

    eprintln!(
        "QMV comparison: Q{bits} g{group_size} BF16, {input_dim}x{output_dim}, M=1..32, {} kernels",
        kernels.len()
    );

    eprintln!("Global warmup...");
    for &(_, force) in &kernels {
        for &batch in &[1, 8, 16] {
            let shape = TestShape {
                batch,
                input_dim,
                output_dim,
            };
            let _ = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, force);
        }
    }
    eprintln!("Warmup done, starting benchmark");

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::with_template("{spinner} {pos} benchmarks [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::new();
    for &batch in &batches {
        let shape = TestShape {
            batch,
            input_dim,
            output_dim,
        };
        for &(name, force) in &kernels {
            progress_bar.set_message(format!("M={batch} {name}"));
            let result = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, force);
            results.push(result);
            progress_bar.inc(1);
        }
    }

    progress_bar.finish_with_message("done");
    print_results_table(&results);
    print_comparison_table(&results, &batches);

    if let Ok(dir) = std::env::var("UZU_TEST_RESULTS_DIR") {
        let path = std::path::Path::new(&dir);
        std::fs::create_dir_all(path).expect("create results dir");
        let file = path.join("quant_matmul_perf.json");
        let wrapper = serde_json::json!({ "device": context.device.name(), "results": results });
        let json = serde_json::to_string_pretty(&wrapper).expect("serialize");
        std::fs::write(&file, json).expect("write results");
        eprintln!("Results written to {}", file.display());
    }
}

/// Compare QmvFast, QmmSmall, and Qmm64x64 for M=1..64 on K=4096, N=14336, Q4 g128 BF16.
#[test]
#[ignore]
fn quant_matmul_perf_m64() {
    let context = Ctx::new().expect("Metal context required");

    let data_type = DataType::BF16;
    let bits: usize = 4;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;

    let input_dim = 4096;
    let output_dim = 14336;
    let batches: Vec<usize> = (1..=64).collect();

    let kernels = [
        ("QmvFast", ForceKernel::QmvFast),
        ("QmmSmall", ForceKernel::QmmTransposedSmall),
        ("Qmm64x64", ForceKernel::QmmTransposed64x64),
    ];

    eprintln!("M=1..64 comparison: Q{bits} g{group_size} BF16, {input_dim}x{output_dim}, {} kernels", kernels.len());

    eprintln!("Global warmup...");
    for &(_, force) in &kernels {
        for &batch in &[1, 8, 32, 64] {
            let shape = TestShape {
                batch,
                input_dim,
                output_dim,
            };
            let _ = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, force);
        }
    }
    eprintln!("Warmup done, starting benchmark");

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::with_template("{spinner} {pos} benchmarks [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::new();
    for &batch in &batches {
        let shape = TestShape {
            batch,
            input_dim,
            output_dim,
        };
        for &(name, force) in &kernels {
            progress_bar.set_message(format!("M={batch} {name}"));
            let result = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, force);
            results.push(result);
            progress_bar.inc(1);
        }
    }

    progress_bar.finish_with_message("done");
    print_results_table(&results);
    print_comparison_table(&results, &batches);

    if let Ok(dir) = std::env::var("UZU_TEST_RESULTS_DIR") {
        let path = std::path::Path::new(&dir);
        std::fs::create_dir_all(path).expect("create results dir");
        let file = path.join("quant_matmul_perf_m64.json");
        let wrapper = serde_json::json!({ "device": context.device.name(), "results": results });
        let json = serde_json::to_string_pretty(&wrapper).expect("serialize");
        std::fs::write(&file, json).expect("write results");
        eprintln!("Results written to {}", file.display());
    }
}

/// Statistical benchmark: run M=1..8 N_REPEATS times per kernel, report min/median/mean/max.
#[test]
#[ignore]
fn quant_matmul_perf_stats() {
    let context = Ctx::new().expect("Metal context required");

    let data_type = DataType::BF16;
    let bits: usize = 4;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;
    let input_dim = 4096;
    let output_dim = 14336;
    let batches: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let n_repeats = 30;

    let kernels = [("QmvFast", ForceKernel::QmvFast), ("QmmSmall", ForceKernel::QmmTransposedSmall)];

    eprintln!("Statistical benchmark: M=1..8, {} kernels × {} repeats", kernels.len(), n_repeats);

    // Aggressive global warmup
    eprintln!("Global warmup...");
    for _ in 0..3 {
        for &(_, force) in &kernels {
            for &batch in &batches {
                let shape = TestShape {
                    batch,
                    input_dim,
                    output_dim,
                };
                let _ = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, force);
            }
        }
    }
    eprintln!("Warmup done.");

    // samples[kernel_idx][m_idx] = Vec<f64>
    let mut samples: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); batches.len()]; kernels.len()];

    for rep in 0..n_repeats {
        if rep % 10 == 0 {
            eprintln!("Repeat {}/{}", rep + 1, n_repeats);
        }
        // Interleave kernels and M values to average out thermal drift
        for (ki, &(_, force)) in kernels.iter().enumerate() {
            for (mi, &batch) in batches.iter().enumerate() {
                let shape = TestShape {
                    batch,
                    input_dim,
                    output_dim,
                };
                let result = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, force);
                if result.status == "ok" {
                    samples[ki][mi].push(result.duration_ms);
                }
            }
        }
    }

    // Print stats table: min / median / mean / max
    println!("\n=== Statistical results (ms): min / median / mean / max ===");
    println!("{:<18} | {:<8} | {:<22} | {:<22} | {:<22} | {:<22}", "Kernel", "M", "Min", "Median", "Mean", "Max");
    println!("{}", "-".repeat(130));

    for (ki, (name, _)) in kernels.iter().enumerate() {
        for (mi, &batch) in batches.iter().enumerate() {
            let mut s = samples[ki][mi].clone();
            if s.is_empty() {
                continue;
            }
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let min = s[0];
            let max = s[s.len() - 1];
            let median = s[s.len() / 2];
            let mean = s.iter().sum::<f64>() / s.len() as f64;
            println!(
                "{:<18} | M={:<6} | {:<22.3} | {:<22.3} | {:<22.3} | {:<22.3}",
                name, batch, min, median, mean, max
            );
        }
    }

    // Print per-M best kernel (by median)
    println!("\n=== Best kernel per M (by median) ===");
    for (mi, &batch) in batches.iter().enumerate() {
        let mut best_idx = 0;
        let mut best_median = f64::INFINITY;
        for (ki, _) in kernels.iter().enumerate() {
            let mut s = samples[ki][mi].clone();
            if s.is_empty() {
                continue;
            }
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = s[s.len() / 2];
            if median < best_median {
                best_median = median;
                best_idx = ki;
            }
        }
        println!("M={}: {} @ {:.3}ms", batch, kernels[best_idx].0, best_median);
    }
}

/// Run QmmSmall only, multiple iterations per M, for xctrace counter capture.
/// Run: cd tools/gpu_trace && uv run gpu_trace run --gpu-counters -- \
///   /path/to/target/release/deps/performance-<hash> \
///   quant_matmul::main::quant_matmul_qmm_small_only --exact --nocapture --ignored
#[test]
#[ignore]
fn quant_matmul_qmm_small_only() {
    let context = Ctx::new().expect("Metal context required");
    let data_type = DataType::BF16;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;
    let bits: usize = 4;
    let input_dim = 4096;
    let output_dim = 14336;
    let batches = [1, 2, 4, 8, 16, 32];

    // Warmup
    for &batch in &batches {
        let shape = TestShape {
            batch,
            input_dim,
            output_dim,
        };
        let _ = bench::benchmark_single(
            &context,
            data_type,
            &shape,
            group_size,
            bits,
            mode,
            ForceKernel::QmmTransposedSmall,
        );
    }

    // Measured: enough iterations for counter stability
    for &batch in &batches {
        let shape = TestShape {
            batch,
            input_dim,
            output_dim,
        };
        for _ in 0..30 {
            let result = bench::benchmark_single(
                &context,
                data_type,
                &shape,
                group_size,
                bits,
                mode,
                ForceKernel::QmmTransposedSmall,
            );
            eprintln!("QmmSmall M={batch}: {:.3}ms ({:.1} GFLOPS)", result.duration_ms, result.gflops);
        }
    }
}

/// Capture a .gputrace of QmvFast at M=1..4 for Xcode analysis, for direct
/// comparison against the QmmSmall trace at the same M values and shape.
///
/// Run: METAL_CAPTURE_ENABLED=1 cargo test --release -p uzu --test performance -- \
///   quant_matmul::main::quant_matmul_capture_qmv_fast_m1_m4 --exact --nocapture --ignored
#[test]
#[ignore]
fn quant_matmul_capture_qmv_fast_m1_m4() {
    use std::path::PathBuf;

    let context = Ctx::new().expect("Metal context required");
    let data_type = DataType::BF16;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;
    let bits: usize = 4;
    let input_dim = 4096;
    let output_dim = 14336;
    let batches = [1, 2, 3, 4];

    let trace_path = PathBuf::from("./traces/qmv_fast_m1_m4.gputrace");
    if trace_path.exists() {
        std::fs::remove_dir_all(&trace_path).ok();
    }
    std::fs::create_dir_all("./traces").ok();

    // Warmup without capture to ramp clocks + warm pipeline caches
    for _ in 0..3 {
        for &batch in &batches {
            let shape = TestShape {
                batch,
                input_dim,
                output_dim,
            };
            let _ = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, ForceKernel::QmvFast);
        }
    }

    context.start_capture(&trace_path).expect("start capture");

    for &batch in &batches {
        let shape = TestShape {
            batch,
            input_dim,
            output_dim,
        };
        let result = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, ForceKernel::QmvFast);
        eprintln!("QmvFast M={batch}: {:.3}ms ({:.1} GFLOPS)", result.duration_ms, result.gflops);
    }

    context.stop_capture().expect("stop capture");
    eprintln!("GPU trace saved to: {}", trace_path.display());
}

/// Capture a .gputrace of QmvFast at M=1..8 for Xcode analysis.
///
/// Run: METAL_CAPTURE_ENABLED=1 cargo test --release -p uzu --test performance -- \
///   quant_matmul::main::quant_matmul_capture_qmv_fast_m1_m8 --exact --nocapture --ignored
#[test]
#[ignore]
fn quant_matmul_capture_qmv_fast_m1_m8() {
    use std::path::PathBuf;

    let context = Ctx::new().expect("Metal context required");
    let data_type = DataType::BF16;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;
    let bits: usize = 4;
    let input_dim = 4096;
    let output_dim = 14336;
    let batches: Vec<usize> = (1..=8).collect();

    let trace_path = PathBuf::from("./traces/qmv_fast_m1_m8.gputrace");
    if trace_path.exists() {
        std::fs::remove_dir_all(&trace_path).ok();
    }
    std::fs::create_dir_all("./traces").ok();

    for _ in 0..3 {
        for &batch in &batches {
            let shape = TestShape {
                batch,
                input_dim,
                output_dim,
            };
            let _ = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, ForceKernel::QmvFast);
        }
    }

    context.start_capture(&trace_path).expect("start capture");

    for &batch in &batches {
        let shape = TestShape {
            batch,
            input_dim,
            output_dim,
        };
        let result = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, ForceKernel::QmvFast);
        eprintln!("QmvFast M={batch}: {:.3}ms ({:.1} GFLOPS)", result.duration_ms, result.gflops);
    }

    context.stop_capture().expect("stop capture");
    eprintln!("GPU trace saved to: {}", trace_path.display());
}

fn capture_qmv_fast_single_m(
    target_m: usize,
    trace_name: &str,
) {
    use std::path::PathBuf;

    let context = Ctx::new().expect("Metal context required");
    let data_type = DataType::BF16;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;
    let bits: usize = 4;
    let input_dim = 4096;
    let output_dim = 14336;

    let trace_path = PathBuf::from(format!("./traces/{trace_name}"));
    if trace_path.exists() {
        std::fs::remove_dir_all(&trace_path).ok();
    }
    std::fs::create_dir_all("./traces").ok();

    let shape = TestShape {
        batch: target_m,
        input_dim,
        output_dim,
    };

    // Warmup without capture to ramp clocks + warm pipeline caches. Warmup the
    // full M=1..4 sweep so pipeline cache state matches the multi-M capture.
    for _ in 0..3 {
        for &batch in &[1, 2, 3, 4] {
            let warmup_shape = TestShape {
                batch,
                input_dim,
                output_dim,
            };
            let _ = bench::benchmark_single(
                &context,
                data_type,
                &warmup_shape,
                group_size,
                bits,
                mode,
                ForceKernel::QmvFast,
            );
        }
    }

    context.start_capture(&trace_path).expect("start capture");

    let result = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, ForceKernel::QmvFast);
    eprintln!("QmvFast M={target_m}: {:.3}ms ({:.1} GFLOPS)", result.duration_ms, result.gflops);

    context.stop_capture().expect("stop capture");
    eprintln!("GPU trace saved to: {}", trace_path.display());
}

/// Capture a .gputrace of QmvFast at M=1 only, so Xcode's cost graph averages
/// purely M=1 dispatches (no mixing with M=2..4).
///
/// Run: METAL_CAPTURE_ENABLED=1 cargo test --release -p uzu --test performance -- \
///   quant_matmul::main::quant_matmul_capture_qmv_fast_m1_only --exact --nocapture --ignored
#[test]
#[ignore]
fn quant_matmul_capture_qmv_fast_m1_only() {
    capture_qmv_fast_single_m(1, "qmv_fast_m1_only.gputrace");
}

/// Capture a .gputrace of QmvFast at M=4 only, so Xcode's cost graph averages
/// purely M=4 dispatches (no mixing with M=1..3).
///
/// Run: METAL_CAPTURE_ENABLED=1 cargo test --release -p uzu --test performance -- \
///   quant_matmul::main::quant_matmul_capture_qmv_fast_m4_only --exact --nocapture --ignored
#[test]
#[ignore]
fn quant_matmul_capture_qmv_fast_m4_only() {
    capture_qmv_fast_single_m(4, "qmv_fast_m4_only.gputrace");
}

/// Capture a .gputrace of QmmSmall at M=1..4 for Xcode analysis of why QmmSmall
/// flatlines around ~0.53 ms regardless of token count in this range.
///
/// Run: METAL_CAPTURE_ENABLED=1 cargo test --release -p uzu --test performance -- \
///   quant_matmul::main::quant_matmul_capture_qmm_small_m1_m4 --exact --nocapture --ignored
#[test]
#[ignore]
fn quant_matmul_capture_qmm_small_m1_m4() {
    use std::path::PathBuf;

    let context = Ctx::new().expect("Metal context required");
    let data_type = DataType::BF16;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;
    let bits: usize = 4;
    let input_dim = 4096;
    let output_dim = 14336;
    let batches = [1, 2, 3, 4];

    let trace_path = PathBuf::from("./traces/qmm_small_m1_m4.gputrace");
    if trace_path.exists() {
        std::fs::remove_dir_all(&trace_path).ok();
    }
    std::fs::create_dir_all("./traces").ok();

    // Warmup without capture to ramp clocks + warm pipeline caches
    for _ in 0..3 {
        for &batch in &batches {
            let shape = TestShape {
                batch,
                input_dim,
                output_dim,
            };
            let _ = bench::benchmark_single(
                &context,
                data_type,
                &shape,
                group_size,
                bits,
                mode,
                ForceKernel::QmmTransposedSmall,
            );
        }
    }

    context.start_capture(&trace_path).expect("start capture");

    for &batch in &batches {
        let shape = TestShape {
            batch,
            input_dim,
            output_dim,
        };
        let result = bench::benchmark_single(
            &context,
            data_type,
            &shape,
            group_size,
            bits,
            mode,
            ForceKernel::QmmTransposedSmall,
        );
        eprintln!("QmmSmall M={batch}: {:.3}ms ({:.1} GFLOPS)", result.duration_ms, result.gflops);
    }

    context.stop_capture().expect("stop capture");
    eprintln!("GPU trace saved to: {}", trace_path.display());
}

/// Benchmark QmvFast, QmmSmall, and QmmSmallSplitK for M=1..8
/// across multiple output dimensions (N=14336, 4096, 1536) to test where
/// split-K helps with GPU utilization.
///
/// Run: cargo test --release -p uzu --test performance -- \
///   quant_matmul::main::quant_matmul_perf_splitk --exact --nocapture --ignored
#[test]
#[ignore]
fn quant_matmul_perf_splitk() {
    let context = Ctx::new().expect("Metal context required");

    let data_type = DataType::BF16;
    let bits: usize = 4;
    let group_size: usize = 128;
    let mode = QuantizationMode::UINT4;

    let input_dim = 4096;
    let output_dims = [14336, 4096, 1536];
    let batches: Vec<usize> = (1..=8).collect();

    let kernels = [
        ("QmvFast", ForceKernel::QmvFast),
        ("QmmSmall", ForceKernel::QmmTransposedSmall),
        ("QmmSmallSplitK", ForceKernel::QmmTransposedSmallSplitK),
    ];

    for &output_dim in &output_dims {
        let tgs = ((output_dim + 31) / 32) * ((1 + 7) / 8);
        let split_k_est = std::cmp::max(1, 512 / tgs);
        eprintln!(
            "\n=== N={output_dim}, K={input_dim} (TGs/M≈{}, est split_k≈{}) ===",
            (output_dim + 31) / 32,
            split_k_est
        );

        // Warmup
        for &(_, force) in &kernels {
            for &batch in &[1, 4, 8] {
                let shape = TestShape {
                    batch,
                    input_dim,
                    output_dim,
                };
                let _ = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, force);
            }
        }

        let mut results = Vec::new();
        for &batch in &batches {
            let shape = TestShape {
                batch,
                input_dim,
                output_dim,
            };
            for &(name, force) in &kernels {
                let result = bench::benchmark_single(&context, data_type, &shape, group_size, bits, mode, force);
                results.push(result);
            }
        }

        print_comparison_table(&results, &batches);
    }
}
