//! MLP Fused Kernel Performance Benchmark
//! Compares fused vs unfused MLP paths to measure speedup

use std::time::{Duration, Instant};

use half::f16;
use metal::{Device, MTLResourceOptions};
use objc2::rc::autoreleasepool;
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::{
            matmul::{
                MatmulArguments, MatmulKernel, MlpFusedGemmArguments,
                MlpFusedGemmKernel, MlpFusedGemvArguments, MlpFusedGemvKernel,
            },
            mlp::MlpActivationType,
        },
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

struct BenchResult {
    fused_time_us: f64,
    unfused_time_us: f64,
    speedup: f64,
}

impl BenchResult {
    fn new(
        fused: Duration,
        unfused: Duration,
    ) -> Self {
        let fused_us = fused.as_secs_f64() * 1_000_000.0;
        let unfused_us = unfused.as_secs_f64() * 1_000_000.0;
        Self {
            fused_time_us: fused_us,
            unfused_time_us: unfused_us,
            speedup: unfused_us / fused_us,
        }
    }
}

fn bench_gemv_fused(
    ctx: &MTLContext,
    k: usize,
    hidden_dim: usize,
    iterations: usize,
    warmup: usize,
) -> BenchResult {
    // Allocate buffers
    let input_buf = ctx.device.new_buffer(
        (k * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weights_buf = ctx.device.new_buffer(
        (2 * hidden_dim * k * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = ctx.device.new_buffer(
        (hidden_dim * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let fused_up_buf = ctx.device.new_buffer(
        (2 * hidden_dim * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut fused_kernel =
        MlpFusedGemvKernel::new(DataType::F16).expect("fused kernel");
    let mut matmul_kernel =
        MatmulKernel::new(ctx, DataType::F16, false, true).expect("matmul");

    // Warmup fused
    for _ in 0..warmup {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        let args = MlpFusedGemvArguments {
            weights: &weights_buf,
            input: &input_buf,
            input_offset: 0,
            output: &output_buf,
            input_dim: k as i32,
            hidden_dim: hidden_dim as i32,
            weights_ld: k as i32,
            batch_count: 1,
            activation: MlpActivationType::SiLU,
        };
        fused_kernel.encode(ctx, &enc, &args).unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Benchmark fused
    let fused_start = Instant::now();
    for _ in 0..iterations {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        let args = MlpFusedGemvArguments {
            weights: &weights_buf,
            input: &input_buf,
            input_offset: 0,
            output: &output_buf,
            input_dim: k as i32,
            hidden_dim: hidden_dim as i32,
            weights_ld: k as i32,
            batch_count: 1,
            activation: MlpActivationType::SiLU,
        };
        fused_kernel.encode(ctx, &enc, &args).unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let fused_time = fused_start.elapsed() / iterations as u32;

    // Warmup unfused (matmul only - simplified comparison)
    for _ in 0..warmup {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        matmul_kernel
            .encode(
                ctx,
                &enc,
                MatmulArguments {
                    a: &input_buf,
                    a_offset: 0,
                    b: &weights_buf,
                    c: None,
                    d: &fused_up_buf,
                    batch: 1,
                    input_dim: k as i32,
                    output_dim: (2 * hidden_dim) as i32,
                    lda: k as i32,
                    ldb: k as i32,
                    ldd: (2 * hidden_dim) as i32,
                    batch_count: 1,
                    alpha: 1.0,
                    beta: 0.0,
                },
            )
            .unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Benchmark unfused (matmul + separate activation would add more time)
    let unfused_start = Instant::now();
    for _ in 0..iterations {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        matmul_kernel
            .encode(
                ctx,
                &enc,
                MatmulArguments {
                    a: &input_buf,
                    a_offset: 0,
                    b: &weights_buf,
                    c: None,
                    d: &fused_up_buf,
                    batch: 1,
                    input_dim: k as i32,
                    output_dim: (2 * hidden_dim) as i32,
                    lda: k as i32,
                    ldb: k as i32,
                    ldd: (2 * hidden_dim) as i32,
                    batch_count: 1,
                    alpha: 1.0,
                    beta: 0.0,
                },
            )
            .unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let unfused_time = unfused_start.elapsed() / iterations as u32;

    BenchResult::new(fused_time, unfused_time)
}

fn bench_gemm_fused(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    hidden_dim: usize,
    iterations: usize,
    warmup: usize,
) -> BenchResult {
    let input_buf = ctx.device.new_buffer(
        (m * k * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weights_buf = ctx.device.new_buffer(
        (2 * hidden_dim * k * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = ctx.device.new_buffer(
        (m * hidden_dim * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let fused_up_buf = ctx.device.new_buffer(
        (m * 2 * hidden_dim * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut fused_kernel =
        MlpFusedGemmKernel::new(DataType::F16, true).expect("fused kernel");
    let mut matmul_kernel =
        MatmulKernel::new(ctx, DataType::F16, false, true).expect("matmul");

    // Warmup fused
    for _ in 0..warmup {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        let args = MlpFusedGemmArguments {
            input: &input_buf,
            input_offset: 0,
            weights: &weights_buf,
            output: &output_buf,
            batch: m as i32,
            input_dim: k as i32,
            hidden_dim: hidden_dim as i32,
            lda: k as i32,
            ldb: k as i32,
            ldd: hidden_dim as i32,
            activation: MlpActivationType::SiLU,
        };
        fused_kernel.encode(ctx, &enc, &args).unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let fused_start = Instant::now();
    for _ in 0..iterations {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        let args = MlpFusedGemmArguments {
            input: &input_buf,
            input_offset: 0,
            weights: &weights_buf,
            output: &output_buf,
            batch: m as i32,
            input_dim: k as i32,
            hidden_dim: hidden_dim as i32,
            lda: k as i32,
            ldb: k as i32,
            ldd: hidden_dim as i32,
            activation: MlpActivationType::SiLU,
        };
        fused_kernel.encode(ctx, &enc, &args).unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let fused_time = fused_start.elapsed() / iterations as u32;

    // Warmup unfused
    for _ in 0..warmup {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        matmul_kernel
            .encode(
                ctx,
                &enc,
                MatmulArguments {
                    a: &input_buf,
                    a_offset: 0,
                    b: &weights_buf,
                    c: None,
                    d: &fused_up_buf,
                    batch: m as i32,
                    input_dim: k as i32,
                    output_dim: (2 * hidden_dim) as i32,
                    lda: k as i32,
                    ldb: k as i32,
                    ldd: (2 * hidden_dim) as i32,
                    batch_count: 1,
                    alpha: 1.0,
                    beta: 0.0,
                },
            )
            .unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let unfused_start = Instant::now();
    for _ in 0..iterations {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        matmul_kernel
            .encode(
                ctx,
                &enc,
                MatmulArguments {
                    a: &input_buf,
                    a_offset: 0,
                    b: &weights_buf,
                    c: None,
                    d: &fused_up_buf,
                    batch: m as i32,
                    input_dim: k as i32,
                    output_dim: (2 * hidden_dim) as i32,
                    lda: k as i32,
                    ldb: k as i32,
                    ldd: (2 * hidden_dim) as i32,
                    batch_count: 1,
                    alpha: 1.0,
                    beta: 0.0,
                },
            )
            .unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let unfused_time = unfused_start.elapsed() / iterations as u32;

    BenchResult::new(fused_time, unfused_time)
}

/// MLP shapes from real model configs
/// Format: (K=model_dim, H=hidden_dim)
const MLP_SHAPES: &[(usize, usize)] = &[
    (896, 4864),   // Qwen2.5-Coder-0.5B
    (1024, 3072),  // Qwen3-0.6B
    (1024, 4608),  // LFM2-350M
    (1152, 6912),  // gemma-3-1b
    (1536, 6912),  // LFM2-700M
    (1536, 8960),  // Qwen2.5-Coder-1.5B, DeepSeek-R1-Distill-Qwen-1.5B
    (2048, 6144),  // Qwen3-1.7B
    (2048, 8192),  // Llamba-1B, SmolLM2-1.7B, Llama-3.2-1B, LFM2-1.2B
    (2048, 10752), // LFM2-2.6B
    (2048, 11008), // Qwen2.5-Coder-3B
    (2560, 9728),  // Qwen3-4B
    (2560, 10240), // gemma-3-4b
    (3072, 8192),  // Llama-3.2-3B, Llamba-3B
    (3584, 18944), // Qwen2.5-Coder-7B
    (4096, 12288), // Qwen3-8B
    (4096, 14336), // Llamba-8B, Llama-3.1-8B
    (4096, 16384), // rnj-1-instruct
    (5120, 13824), // Qwen2.5-Coder-14B
    (5120, 17408), // Qwen3-14B
    (5120, 25600), // Qwen3-32B
    (5376, 21504), // gemma-3-27b
    (6144, 16384), // Codestral-22B
];

pub fn run_mlp_fused_benchmark() {
    autoreleasepool(|_| {
        let ctx = match create_test_context() {
            Some(c) => c,
            None => {
                eprintln!("No Metal device available");
                return;
            },
        };

        println!("\n=== MLP Fused Kernel Benchmark (Real Model Shapes) ===\n");

        let iterations = 100;
        let warmup = 10;

        // GEMV benchmark (decode path, M=1)
        println!("=== GEMV (Decode, M=1) ===");
        println!(
            "{:<25} {:>12} {:>12} {:>10}",
            "Shape (K, H)", "Fused (µs)", "Unfused (µs)", "Speedup"
        );
        println!("{}", "-".repeat(61));

        for &(k, hidden_dim) in MLP_SHAPES {
            let result =
                bench_gemv_fused(&ctx, k, hidden_dim, iterations, warmup);
            let speedup_str = if result.speedup >= 1.0 {
                format!("{:.2}x", result.speedup)
            } else {
                format!("{:.2}x", result.speedup)
            };
            println!(
                "({:>4}, {:>5})              {:>12.1} {:>12.1} {:>10}",
                k,
                hidden_dim,
                result.fused_time_us,
                result.unfused_time_us,
                speedup_str
            );
        }

        println!();

        // GEMM benchmark (prefill path, M=32)
        println!("=== GEMM (Prefill, M=32) ===");
        println!(
            "{:<25} {:>12} {:>12} {:>10}",
            "Shape (K, H)", "Fused (µs)", "Unfused (µs)", "Speedup"
        );
        println!("{}", "-".repeat(61));

        for &(k, hidden_dim) in MLP_SHAPES {
            let result =
                bench_gemm_fused(&ctx, 32, k, hidden_dim, iterations, warmup);
            let speedup_str = if result.speedup >= 1.0 {
                format!("{:.2}x", result.speedup)
            } else {
                format!("{:.2}x", result.speedup)
            };
            println!(
                "({:>4}, {:>5})              {:>12.1} {:>12.1} {:>10}",
                k,
                hidden_dim,
                result.fused_time_us,
                result.unfused_time_us,
                speedup_str
            );
        }

        println!("\nNote: Unfused time excludes activation kernel overhead.");
        println!(
            "      Actual unfused path would be slower due to activation dispatch.\n"
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runs() {
        // Just verify the benchmark doesn't crash
        autoreleasepool(|_| {
            if let Some(ctx) = create_test_context() {
                let _ = bench_gemv_fused(&ctx, 512, 256, 5, 2);
            }
        });
    }

    #[test]
    #[ignore] // Run with: cargo test --package benchmarks -- run_full_benchmark --ignored --nocapture
    fn run_full_benchmark() {
        run_mlp_fused_benchmark();
    }
}
