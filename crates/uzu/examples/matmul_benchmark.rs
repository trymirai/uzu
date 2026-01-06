//! Matmul benchmark comparing Metal GEMM kernel vs MPSGraph
//!
//! Run with: cargo run --release -p uzu --example matmul_benchmark

use std::{collections::HashMap, io::Write, time::Instant};

use half::bf16;
use metal::{Device, MTLResourceOptions};
use mpsgraph::{
    CommandBuffer, Device as MPSDevice, ExecutableExecutionDescriptor, Graph,
    ShapedType, TensorData,
};
use objc2::rc::autoreleasepool;
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::{MatmulArguments, MatmulKernel},
    },
};

struct BenchmarkResult {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    metal_ms: f64,
    mpsgraph_ms: f64,
    metal_gflops: f64,
    mpsgraph_gflops: f64,
}

fn compute_gflops(
    m: usize,
    k: usize,
    n: usize,
    time_ms: f64,
) -> f64 {
    let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
    flops / (time_ms * 1e6)
}

fn benchmark_metal_gemm(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    n: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> f64 {
    // A is [M, K] row-major
    let a: Vec<bf16> = (0..(m * k))
        .map(|i| bf16::from_f32(((i % 13) as f32) * 0.01))
        .collect();
    // B is [N, K] row-major (transpose_b = true, like LLM weight matrices)
    let b: Vec<bf16> = (0..(n * k))
        .map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.1))
        .collect();

    let a_buf = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (m * n * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // transpose_b = true: B is stored as [N, K], matching LLM weight layout
    let mut kernel = MatmulKernel::new(ctx, DataType::BF16, false, true)
        .expect("kernel new");

    // Warmup
    for _ in 0..warmup_iters {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        kernel
            .encode(
                ctx,
                &enc,
                MatmulArguments {
                    a: &a_buf,
                    b: &b_buf,
                    d: &d_buf,
                    batch: m as i32,
                    input_dim: k as i32,
                    output_dim: n as i32,
                    lda: k as i32,
                    ldb: k as i32, // B is [N, K], stride = K
                    ldd: n as i32,
                    batch_count: 1,
                },
            )
            .expect("encode");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..bench_iters {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        kernel
            .encode(
                ctx,
                &enc,
                MatmulArguments {
                    a: &a_buf,
                    b: &b_buf,
                    d: &d_buf,
                    batch: m as i32,
                    input_dim: k as i32,
                    output_dim: n as i32,
                    lda: k as i32,
                    ldb: k as i32, // B is [N, K], stride = K
                    ldd: n as i32,
                    batch_count: 1,
                },
            )
            .expect("encode");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let elapsed = start.elapsed();
    (elapsed.as_secs_f64() * 1000.0) / bench_iters as f64
}

fn benchmark_metal_gemv(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    n: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> f64 {
    // GEMV requires transpose_b = true (B stored as [N, K]).
    let a: Vec<bf16> = (0..(m * k))
        .map(|i| bf16::from_f32(((i % 13) as f32) * 0.01))
        .collect();
    let b: Vec<bf16> = (0..(n * k))
        .map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.1))
        .collect();

    let a_buf = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (m * n * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut kernel = MatmulKernel::new(ctx, DataType::BF16, false, true)
        .expect("kernel new");

    for _ in 0..warmup_iters {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        kernel
            .encode(
                ctx,
                &enc,
                MatmulArguments {
                    a: &a_buf,
                    b: &b_buf,
                    d: &d_buf,
                    batch: m as i32,
                    input_dim: k as i32,
                    output_dim: n as i32,
                    lda: k as i32,
                    ldb: k as i32,
                    ldd: n as i32,
                    batch_count: 1,
                },
            )
            .expect("encode");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let start = Instant::now();
    for _ in 0..bench_iters {
        let cb = ctx.command_queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        kernel
            .encode(
                ctx,
                &enc,
                MatmulArguments {
                    a: &a_buf,
                    b: &b_buf,
                    d: &d_buf,
                    batch: m as i32,
                    input_dim: k as i32,
                    output_dim: n as i32,
                    lda: k as i32,
                    ldb: k as i32,
                    ldd: n as i32,
                    batch_count: 1,
                },
            )
            .expect("encode");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let elapsed = start.elapsed();
    (elapsed.as_secs_f64() * 1000.0) / bench_iters as f64
}

fn benchmark_mpsgraph_matmul(
    device: &metal::Device,
    m: usize,
    k: usize,
    n: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> f64 {
    autoreleasepool(|_| {
        print!("graph ");
        let _ = std::io::stdout().flush();
        let graph = Graph::new();

        // Create input placeholders
        let a_shape = [m as isize, k as isize];
        let b_shape = [k as isize, n as isize];

        let a_placeholder = graph.placeholder(
            Some(&a_shape),
            mpsgraph::DataType::BFloat16,
            Some("a"),
        );
        let b_placeholder = graph.placeholder(
            Some(&b_shape),
            mpsgraph::DataType::BFloat16,
            Some("b"),
        );

        // Matrix multiplication
        let result =
            graph.matrix_multiplication(&a_placeholder, &b_placeholder, None);

        // Create shaped types for compilation
        let a_shaped_type = ShapedType::new_with_shape_data_type(
            Some(&a_shape),
            mpsgraph::DataType::BFloat16,
        );
        let b_shaped_type = ShapedType::new_with_shape_data_type(
            Some(&b_shape),
            mpsgraph::DataType::BFloat16,
        );

        let feeds = HashMap::from([
            (&*a_placeholder, &*a_shaped_type),
            (&*b_placeholder, &*b_shaped_type),
        ]);

        // Compile
        print!("compile ");
        let _ = std::io::stdout().flush();
        let mps_device = MPSDevice::with_device(device);
        let executable =
            graph.compile(&mps_device, &feeds, &[&result], None, None);
        print!("ok ");

        // Create execution descriptor
        let execution_descriptor = ExecutableExecutionDescriptor::new();

        // Create input data buffers
        let a: Vec<bf16> = (0..(m * k))
            .map(|i| bf16::from_f32(((i % 13) as f32) * 0.01))
            .collect();
        let b: Vec<bf16> = (0..(k * n))
            .map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.1))
            .collect();

        let a_buf = device.new_buffer_with_data(
            a.as_ptr() as *const _,
            (a.len() * core::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = device.new_buffer_with_data(
            b.as_ptr() as *const _,
            (b.len() * core::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create output buffer
        let output_buf = device.new_buffer(
            (m * n * core::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_queue = device.new_command_queue();

        // Create tensor data wrappers
        let a_tensor_data = TensorData::new_with_mtl_buffer(
            &a_buf,
            &[m, k],
            mpsgraph::DataType::BFloat16,
            None,
        );
        let b_tensor_data = TensorData::new_with_mtl_buffer(
            &b_buf,
            &[k, n],
            mpsgraph::DataType::BFloat16,
            None,
        );
        let output_tensor_data = TensorData::new_with_mtl_buffer(
            &output_buf,
            &[m, n],
            mpsgraph::DataType::BFloat16,
            None,
        );

        let inputs: &[&TensorData] = &[&a_tensor_data, &b_tensor_data];
        let outputs: &[&TensorData] = &[&output_tensor_data];

        // Warmup
        execution_descriptor.set_enable_commit_and_continue(true);
        print!("warmup ");
        let _ = std::io::stdout().flush();
        for _ in 0..warmup_iters {
            let command_buffer =
                CommandBuffer::from_command_queue(&command_queue);
            let root_cb = command_buffer.root_command_buffer().to_owned();
            let _ = executable.encode_to_command_buffer(
                &command_buffer,
                inputs,
                Some(outputs),
                Some(&execution_descriptor),
            );
            command_buffer.commit_and_continue();
            root_cb.wait_until_completed();
        }
        print!("ok ");
        let _ = std::io::stdout().flush();

        // Benchmark
        print!("bench ");
        let _ = std::io::stdout().flush();
        let start = Instant::now();
        for _ in 0..bench_iters {
            let command_buffer =
                CommandBuffer::from_command_queue(&command_queue);
            let root_cb = command_buffer.root_command_buffer().to_owned();
            let _ = executable.encode_to_command_buffer(
                &command_buffer,
                inputs,
                Some(outputs),
                Some(&execution_descriptor),
            );
            command_buffer.commit_and_continue();
            root_cb.wait_until_completed();
        }
        let elapsed = start.elapsed();
        print!("done ");
        let _ = std::io::stdout().flush();
        (elapsed.as_secs_f64() * 1000.0) / bench_iters as f64
    })
}

fn run_benchmark(
    ctx: &MTLContext,
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iters: usize,
) -> BenchmarkResult {
    print!("[Metal] ");
    let _ = std::io::stdout().flush();
    let metal_ms = if m <= 8 || n == 1 {
        benchmark_metal_gemv(ctx, m, k, n, warmup, iters)
    } else {
        benchmark_metal_gemm(ctx, m, k, n, warmup, iters)
    };
    print!("{:.2}ms ", metal_ms);
    let _ = std::io::stdout().flush();

    print!("[MPS] ");
    let _ = std::io::stdout().flush();
    let mpsgraph_ms =
        benchmark_mpsgraph_matmul(&ctx.device, m, k, n, warmup, iters);
    print!("{:.2}ms ", mpsgraph_ms);
    let _ = std::io::stdout().flush();

    BenchmarkResult {
        name,
        m,
        k,
        n,
        metal_ms,
        mpsgraph_ms,
        metal_gflops: compute_gflops(m, k, n, metal_ms),
        mpsgraph_gflops: compute_gflops(m, k, n, mpsgraph_ms),
    }
}

fn main() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let ctx = MTLContext::new(device, command_queue)
        .expect("Failed to create MTLContext");

    println!("Metal GEMM vs MPSGraph Matmul Benchmark");
    println!("Device: {}", ctx.device.name());
    println!("========================================\n");

    let warmup = 10;
    let iters = 50;

    // Different tensor shapes to test - typical LLM shapes
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("GEMV small (1 token)", 1, 2048, 2048),
        ("GEMV medium (1 token)", 1, 4096, 4096),
        ("GEMV large (1 token)", 1, 8192, 8192),
        ("Small batch (8 tokens)", 8, 2048, 2048),
        ("Medium batch (32 tokens)", 32, 2048, 2048),
        ("Large batch (128 tokens)", 128, 2048, 2048),
        ("Prefill small (256 tokens)", 256, 2048, 2048),
        ("Prefill medium (512 tokens)", 512, 2048, 2048),
        ("Prefill large (1024 tokens)", 1024, 2048, 2048),
        ("Square small", 256, 256, 256),
        ("Square medium", 512, 512, 512),
        ("Square large", 1024, 1024, 1024),
        ("MLP up (1 token)", 1, 2048, 8192),
        ("MLP down (1 token)", 1, 8192, 2048),
        ("MLP up (128 tokens)", 128, 2048, 8192),
        ("MLP down (128 tokens)", 128, 8192, 2048),
    ];

    println!(
        "{:<30} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Shape",
        "M",
        "K",
        "N",
        "Metal(ms)",
        "MPS(ms)",
        "Metal(GF)",
        "MPS(GF)",
        "Speedup"
    );
    println!("{}", "-".repeat(110));

    for (name, m, k, n) in shapes {
        // Print progress indicator
        print!("Running {}... ", name);
        let _ = std::io::stdout().flush();

        let result = run_benchmark(&ctx, name, *m, *k, *n, warmup, iters);
        // Speedup = MPS time / Metal time. >1.0 means Metal is faster.
        let speedup = result.mpsgraph_ms / result.metal_ms;
        let speedup_str = if speedup >= 1.0 {
            format!("{:.2}x ✓", speedup)
        } else {
            format!("{:.2}x", speedup)
        };

        // Clear line and print result
        print!("\r");
        println!(
            "{:<30} {:>6} {:>6} {:>6} {:>10.4} {:>10.4} {:>10.1} {:>10.1} {:>8}",
            result.name,
            result.m,
            result.k,
            result.n,
            result.metal_ms,
            result.mpsgraph_ms,
            result.metal_gflops,
            result.mpsgraph_gflops,
            speedup_str
        );
        let _ = std::io::stdout().flush();
    }

    println!("\n=== Summary ===");
    println!("Speedup = MPS time / Metal time");
    println!("Speedup > 1.0 means Metal is faster ✓");
    println!("Speedup < 1.0 means MPS is faster");
}
