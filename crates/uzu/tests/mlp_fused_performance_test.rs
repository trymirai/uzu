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
            matmul::{MatmulArguments, MatmulKernel},
            mlp::MlpActivationType,
            mlp_fused::{MlpFusedArguments, MlpFusedKernel},
        },
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

struct BenchmarkResult {
    fused_time_microseconds: f64,
    unfused_time_microseconds: f64,
    speedup: f64,
}

impl BenchmarkResult {
    fn new(
        fused_duration: Duration,
        unfused_duration: Duration,
    ) -> Self {
        let fused_microseconds = fused_duration.as_secs_f64() * 1_000_000.0;
        let unfused_microseconds = unfused_duration.as_secs_f64() * 1_000_000.0;
        Self {
            fused_time_microseconds: fused_microseconds,
            unfused_time_microseconds: unfused_microseconds,
            speedup: unfused_microseconds / fused_microseconds,
        }
    }
}

fn benchmark_gemv_fused(
    context: &MTLContext,
    input_dimension: usize,
    hidden_dimension: usize,
    iteration_count: usize,
    warmup_count: usize,
) -> BenchmarkResult {
    let input_buffer = context.device.new_buffer(
        (input_dimension * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weights_buffer = context.device.new_buffer(
        (2 * hidden_dimension * input_dimension * std::mem::size_of::<f16>())
            as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = context.device.new_buffer(
        (hidden_dimension * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let fused_up_buffer = context.device.new_buffer(
        (2 * hidden_dimension * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut fused_kernel =
        MlpFusedKernel::new(DataType::F16, true).expect("fused kernel");
    let mut matmul_kernel = MatmulKernel::new(DataType::F16).expect("matmul");

    let fused_args = MlpFusedArguments {
        input: &input_buffer,
        input_offset: 0,
        weights: &weights_buffer,
        output: &output_buffer,
        batch: 1,
        input_dim: input_dimension as i32,
        hidden_dim: hidden_dimension as i32,
        lda: input_dimension as i32,
        ldb: input_dimension as i32,
        ldd: hidden_dimension as i32,
        batch_count: 1,
        activation: MlpActivationType::SiLU,
    };

    // Warmup fused
    for _ in 0..warmup_count {
        let command_buffer =
            context.command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        fused_kernel
            .encode(context, &encoder, &fused_args)
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Benchmark fused
    let fused_start = Instant::now();
    for _ in 0..iteration_count {
        let command_buffer =
            context.command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        fused_kernel
            .encode(context, &encoder, &fused_args)
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    let fused_duration = fused_start.elapsed() / iteration_count as u32;

    let matmul_args = MatmulArguments {
        a: &input_buffer,
        a_offset: 0,
        b: &weights_buffer,
        c: None,
        d: &fused_up_buffer,
        bias: None,
        batch: 1,
        input_dim: input_dimension as i32,
        output_dim: (2 * hidden_dimension) as i32,
        lda: input_dimension as i32,
        ldb: input_dimension as i32,
        ldd: (2 * hidden_dimension) as i32,
        batch_count: 1,
        alpha: 1.0,
        beta: 0.0,
        transpose_a: false,
        transpose_b: true,
    };

    // Warmup unfused
    for _ in 0..warmup_count {
        let command_buffer =
            context.command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        matmul_kernel
            .encode(context, &encoder, matmul_args.clone())
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Benchmark unfused
    let unfused_start = Instant::now();
    for _ in 0..iteration_count {
        let command_buffer =
            context.command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        matmul_kernel
            .encode(context, &encoder, matmul_args.clone())
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    let unfused_duration = unfused_start.elapsed() / iteration_count as u32;

    BenchmarkResult::new(fused_duration, unfused_duration)
}

fn benchmark_gemm_fused(
    context: &MTLContext,
    batch_size: usize,
    input_dimension: usize,
    hidden_dimension: usize,
    iteration_count: usize,
    warmup_count: usize,
) -> BenchmarkResult {
    let input_buffer = context.device.new_buffer(
        (batch_size * input_dimension * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weights_buffer = context.device.new_buffer(
        (2 * hidden_dimension * input_dimension * std::mem::size_of::<f16>())
            as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = context.device.new_buffer(
        (batch_size * hidden_dimension * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let fused_up_buffer = context.device.new_buffer(
        (batch_size * 2 * hidden_dimension * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut fused_kernel =
        MlpFusedKernel::new(DataType::F16, true).expect("fused kernel");
    let mut matmul_kernel = MatmulKernel::new(DataType::F16).expect("matmul");

    let fused_args = MlpFusedArguments {
        input: &input_buffer,
        input_offset: 0,
        weights: &weights_buffer,
        output: &output_buffer,
        batch: batch_size as i32,
        input_dim: input_dimension as i32,
        hidden_dim: hidden_dimension as i32,
        lda: input_dimension as i32,
        ldb: input_dimension as i32,
        ldd: hidden_dimension as i32,
        batch_count: 1,
        activation: MlpActivationType::SiLU,
    };

    // Warmup fused
    for _ in 0..warmup_count {
        let command_buffer =
            context.command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        fused_kernel
            .encode(context, &encoder, &fused_args)
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    let fused_start = Instant::now();
    for _ in 0..iteration_count {
        let command_buffer =
            context.command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        fused_kernel
            .encode(context, &encoder, &fused_args)
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    let fused_duration = fused_start.elapsed() / iteration_count as u32;

    let matmul_args = MatmulArguments {
        a: &input_buffer,
        a_offset: 0,
        b: &weights_buffer,
        c: None,
        d: &fused_up_buffer,
        bias: None,
        batch: batch_size as i32,
        input_dim: input_dimension as i32,
        output_dim: (2 * hidden_dimension) as i32,
        lda: input_dimension as i32,
        ldb: input_dimension as i32,
        ldd: (2 * hidden_dimension) as i32,
        batch_count: 1,
        alpha: 1.0,
        beta: 0.0,
        transpose_a: false,
        transpose_b: true,
    };

    // Warmup unfused
    for _ in 0..warmup_count {
        let command_buffer =
            context.command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        matmul_kernel
            .encode(context, &encoder, matmul_args.clone())
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    let unfused_start = Instant::now();
    for _ in 0..iteration_count {
        let command_buffer =
            context.command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        matmul_kernel
            .encode(context, &encoder, matmul_args.clone())
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    let unfused_duration = unfused_start.elapsed() / iteration_count as u32;

    BenchmarkResult::new(fused_duration, unfused_duration)
}

/// MLP shapes from real model configs
/// Format: (input_dimension, hidden_dimension)
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

/// Batch sizes to test: decode (1) and prefill sizes
const BATCH_SIZES: &[usize] = &[1, 8, 16, 32, 64, 128, 256, 512];

#[test]
#[ignore] // Run with: cargo test --package uzu --test mlp_fused_performance_test -- --ignored --nocapture
fn mlp_fused_performance_benchmark() {
    autoreleasepool(|_| {
        let context = match create_test_context() {
            Some(context) => context,
            None => {
                eprintln!("No Metal device available");
                return;
            },
        };

        let iteration_count = 100;
        let warmup_count = 10;

        println!(
            "{:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
            "batch_size",
            "input_dim",
            "hidden_dim",
            "Fused (µs)",
            "Unfused (µs)",
            "Speedup"
        );
        println!("{}", "-".repeat(74));

        for &batch_size in BATCH_SIZES {
            for &(input_dimension, hidden_dimension) in MLP_SHAPES {
                let result = if batch_size == 1 {
                    benchmark_gemv_fused(
                        &context,
                        input_dimension,
                        hidden_dimension,
                        iteration_count,
                        warmup_count,
                    )
                } else {
                    benchmark_gemm_fused(
                        &context,
                        batch_size,
                        input_dimension,
                        hidden_dimension,
                        iteration_count,
                        warmup_count,
                    )
                };
                let speedup_string = format!("{:.2}x", result.speedup);
                println!(
                    "{:>12} {:>12} {:>12} {:>12.1} {:>12.1} {:>10}",
                    batch_size,
                    input_dimension,
                    hidden_dimension,
                    result.fused_time_microseconds,
                    result.unfused_time_microseconds,
                    speedup_string
                );
            }
        }
    });
}
