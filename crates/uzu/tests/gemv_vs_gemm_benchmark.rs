//! GEMV vs GEMM Kernel Performance Benchmark
//! Tests at what batch size GEMV becomes slower than GEMM

use std::time::Instant;

use half::bf16;
use metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice,
    MTLDeviceExt, MTLResourceOptions,
};
use objc2::rc::autoreleasepool;
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::matmul::{
            MatmulArguments, MatmulKernel,
            determine_kernel_variant,
        },
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = <dyn MTLDevice>::system_default()?;
    let command_queue = device.new_command_queue()?;
    MTLContext::new(device, command_queue).ok()
}

struct BenchmarkResult {
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
    time_microseconds: f64,
    kernel_variant: String,
}

fn benchmark_matmul(
    context: &MTLContext,
    batch_size: usize,
    input_dim: usize,
    output_dim: usize,
    iteration_count: usize,
    warmup_count: usize,
) -> BenchmarkResult {
    let use_transposed_b = std::env::var("UZU_GEMV_USE_TRANSPOSED_B")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0)
        != 0;

    let a_buffer = context
        .device
        .new_buffer(
            batch_size * input_dim * std::mem::size_of::<bf16>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let b_buffer = context
        .device
        .new_buffer(
            output_dim * input_dim * std::mem::size_of::<bf16>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let d_buffer = context
        .device
        .new_buffer(
            batch_size * output_dim * std::mem::size_of::<bf16>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");

    if use_transposed_b {
        let a_len = batch_size * input_dim;
        let b_len = output_dim * input_dim;
        let mut a_host = vec![bf16::from_f32(0.0); a_len];
        let mut b_host = vec![bf16::from_f32(0.0); b_len];
        for i in 0..a_len {
            a_host[i] = bf16::from_f32((i % 256) as f32);
        }
        for i in 0..b_len {
            b_host[i] = bf16::from_f32((i % 127) as f32);
        }

        let mut bt_host = vec![bf16::from_f32(0.0); b_len];
        for n in 0..output_dim {
            let row_offset = n * input_dim;
            for k in 0..input_dim {
                bt_host[k * output_dim + n] = b_host[row_offset + k];
            }
        }

        unsafe {
            let a_ptr = a_buffer.contents().as_ptr() as *mut bf16;
            std::ptr::copy_nonoverlapping(a_host.as_ptr(), a_ptr, a_len);
            let b_ptr = b_buffer.contents().as_ptr() as *mut bf16;
            std::ptr::copy_nonoverlapping(bt_host.as_ptr(), b_ptr, b_len);
        }
    }

    let mut kernel = MatmulKernel::new(DataType::BF16).expect("kernel");

    let transpose_b = if use_transposed_b { false } else { true };
    let ldb = if transpose_b { input_dim } else { output_dim };

    let args = MatmulArguments {
        a: &a_buffer,
        a_offset: 0,
        b: &b_buffer,
        c: None,
        d: &d_buffer,
        bias: None,
        batch: batch_size as i32,
        input_dim: input_dim as i32,
        output_dim: output_dim as i32,
        lda: input_dim as i32,
        ldb: ldb as i32,
        ldd: output_dim as i32,
        batch_count: 1,
        alpha: 1.0,
        beta: 0.0,
        transpose_a: false,
        transpose_b,
    };

    let kernel_variant = determine_kernel_variant(context, DataType::BF16, &args)
        .map(|v| v.as_str().to_string())
        .unwrap_or_else(|_| "?".to_string());

    // Warmup
    for _ in 0..warmup_count {
        let command_buffer = context
            .command_queue
            .command_buffer()
            .expect("Failed to create command buffer")
            .to_owned();
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute encoder");
        kernel.encode(context, &encoder, args.clone()).unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iteration_count {
        let command_buffer = context
            .command_queue
            .command_buffer()
            .expect("Failed to create command buffer")
            .to_owned();
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute encoder");
        kernel.encode(context, &encoder, args.clone()).unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    let duration = start.elapsed() / iteration_count as u32;
    let time_microseconds = duration.as_secs_f64() * 1_000_000.0;

    BenchmarkResult {
        batch_size,
        input_dim,
        output_dim,
        time_microseconds,
        kernel_variant,
    }
}

/// Model shapes from real configs: (input_dim, output_dim)
const MODEL_SHAPES: &[(usize, usize)] = &[
    // Qwen2.5-Coder-0.5B
    (896, 896),
    (896, 4864),
    (4864, 896),
    // Llama-3.2-1B / SmolLM2-1.7B
    (2048, 2048),
    (2048, 8192),
    (8192, 2048),
    // Llama-3.2-3B
    (3072, 3072),
    (3072, 8192),
    (8192, 3072),
    // Llama-3.1-8B / Qwen3-8B
    (4096, 4096),
    (4096, 6144),
    (4096, 14336),
    (14336, 4096),
    // Qwen2.5-Coder-7B
    (3584, 18944),
    (18944, 3584),
];

/// Batch sizes to test
const BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

#[test]
#[ignore] // Run with: cargo test --package uzu --test gemv_vs_gemm_benchmark -- --ignored --nocapture
fn gemv_vs_gemm_threshold_benchmark() {
    autoreleasepool(|_| {
        let context = match create_test_context() {
            Some(context) => context,
            None => {
                eprintln!("No Metal device available");
                return;
            },
        };

        let iteration_count = 200;
        let warmup_count = 20;

        println!("\n=== GEMV vs GEMM Threshold Benchmark ===\n");
        println!(
            "{:>8} {:>8} {:>8} {:>12} {:>10}",
            "batch", "K", "N", "time (µs)", "kernel"
        );
        println!("{}", "-".repeat(52));

        for &(input_dim, output_dim) in MODEL_SHAPES {
            println!("\n--- K={}, N={} ---", input_dim, output_dim);
            
            for &batch_size in BATCH_SIZES {
                let result = benchmark_matmul(
                    &context,
                    batch_size,
                    input_dim,
                    output_dim,
                    iteration_count,
                    warmup_count,
                );

                println!(
                    "{:>8} {:>8} {:>8} {:>12.2} {:>10}",
                    result.batch_size,
                    result.input_dim,
                    result.output_dim,
                    result.time_microseconds,
                    result.kernel_variant,
                );
            }
        }
    });
}
