use std::time::Instant;

use half::f16;
use metal::{Device, MTLResourceOptions};
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::quant_matmul::{
            QuantizedMatmulArguments, QuantizedMatmulKernel,
        },
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

// Packs 4-bit weights into bytes (4 values per 16-bit word, matching qdot expectations)
fn pack_u4_weights(values: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((values.len() + 3) / 4 * 2); // 4 weights per 16-bit word
    for chunk in values.chunks(4) {
        let w0 = (*chunk.get(0).unwrap_or(&0) as u16) & 0x0F; // Bits 0-3
        let w1 = ((*chunk.get(1).unwrap_or(&0) as u16) & 0x0F) << 4; // Bits 4-7  
        let w2 = ((*chunk.get(2).unwrap_or(&0) as u16) & 0x0F) << 8; // Bits 8-11
        let w3 = ((*chunk.get(3).unwrap_or(&0) as u16) & 0x0F) << 12; // Bits 12-15

        let word: u16 = w0 | w1 | w2 | w3;
        out.push(word as u8); // Low byte
        out.push((word >> 8) as u8); // High byte
    }
    out
}

fn cpu_reference(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],      // A matrix (M x K)
    b_quant: &[u8], // B matrix (N x K), quantized and packed, (N * K / 2) in size
    scales: &[f32], // scales for B (N x ceil(K/group_size))
    biases: &[f32], // biases for B (N x ceil(K/group_size))
) -> Vec<f32> {
    let group_size = 64;
    let num_groups = (k + group_size - 1) / group_size;
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for g in 0..num_groups {
                let scale = scales[j * num_groups + g] as f64;
                let bias = biases[j * num_groups + g] as f64;
                let l_start = g * group_size;
                let l_end = (l_start + group_size).min(k);
                let mut group_acc = 0.0f64;
                let mut group_sum = 0.0f64;
                for l in l_start..l_end {
                    let word_idx = (j * k + l) / 4; // 4 weights per 16-bit word
                    let word_offset = (j * k + l) % 4; // Which nibble in the word
                    let byte_idx = word_idx * 2; // 2 bytes per word

                    let word = if byte_idx + 1 < b_quant.len() {
                        b_quant[byte_idx] as u16
                            | ((b_quant[byte_idx + 1] as u16) << 8)
                    } else {
                        0
                    };

                    let val_q = match word_offset {
                        0 => word & 0x000F,         // Bits 0-3
                        1 => (word & 0x00F0) >> 4,  // Bits 4-7
                        2 => (word & 0x0F00) >> 8,  // Bits 8-11
                        3 => (word & 0xF000) >> 12, // Bits 12-15
                        _ => 0,
                    } as f64;
                    let val_a = a[i * k + l] as f64;
                    group_acc += val_a * val_q;
                    group_sum += val_a;
                }
                acc += scale * group_acc + bias * group_sum;
            }
            y[i * n + j] = acc as f32;
        }
    }
    y
}

fn execute_quantized_matmul(
    ctx: &MTLContext,
    m: usize,
    n: usize,
    k: usize,
    kernel_name: &str,
    iterations: usize,
    validate: bool,
) -> f64 {
    // Prepare deterministic weights: w(row,k) = row+1
    let mut weights_q4: Vec<u8> = Vec::with_capacity(m * k);
    for row in 0..m {
        for _k in 0..k {
            let v = (row as u8 + 1) & 0x0F;
            weights_q4.push(v);
        }
    }
    let weights_packed = pack_u4_weights(&weights_q4);

    let num_groups = (k + 63) / 64;
    let scales: Vec<f16> = vec![f16::from_f32(1.0); m * num_groups];
    let biases: Vec<f16> = vec![f16::from_f32(0.0); m * num_groups];

    // Prepare input data
    let (x_f32, x_f16) = if n == 1 {
        // GEMV case: single input vector
        let x_f32: Vec<f32> = (1..=k).map(|i| i as f32 / k as f32).collect();
        let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
        (x_f32, x_f16)
    } else {
        // GEMM case: multiple input vectors (column-major)
        let mut x_f32: Vec<f32> = Vec::with_capacity(k * n);
        for _col in 0..n {
            for i in 0..k {
                x_f32.push((i + 1) as f32 / k as f32);
            }
        }
        let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
        (x_f32, x_f16)
    };

    // Create buffers
    let w_buf = ctx.device.new_buffer_with_data(
        weights_packed.as_ptr() as *const _,
        (weights_packed.len() * std::mem::size_of::<u8>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let s_buf = ctx.device.new_buffer_with_data(
        scales.as_ptr() as *const _,
        (scales.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        biases.as_ptr() as *const _,
        (biases.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let x_buf = ctx.device.new_buffer_with_data(
        x_f16.as_ptr() as *const _,
        (x_f16.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = ctx.device.new_buffer(
        (m * n * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create kernel
    let kernel =
        QuantizedMatmulKernel::new(&ctx, DataType::F16, kernel_name).unwrap();

    // Warmup (only for benchmarks)
    if iterations > 1 {
        for _ in 0..3 {
            let args = QuantizedMatmulArguments {
                a_buffer: &x_buf,
                b_buffer: &w_buf,
                scales_buffer: &s_buf,
                biases_buffer: &b_buf,
                output_buffer: &y_buf,
                m: n as i32,
                n: m as i32,
                k: k as i32,
            };
            let cb_ref = ctx.command_queue.new_command_buffer();
            let encoder = cb_ref.new_compute_command_encoder();
            kernel.encode(encoder, args).unwrap();
            encoder.end_encoding();
            cb_ref.commit();
            cb_ref.wait_until_completed();
        }
    }

    // Execute and time
    let start = Instant::now();
    for _ in 0..iterations {
        let args = QuantizedMatmulArguments {
            a_buffer: &x_buf,
            b_buffer: &w_buf,
            scales_buffer: &s_buf,
            biases_buffer: &b_buf,
            output_buffer: &y_buf,
            m: n as i32,
            n: m as i32,
            k: k as i32,
        };
        let cb_ref = ctx.command_queue.new_command_buffer();
        let encoder = cb_ref.new_compute_command_encoder();
        kernel.encode(encoder, args).unwrap();
        encoder.end_encoding();
        cb_ref.commit();
        cb_ref.wait_until_completed();
    }
    let elapsed = start.elapsed();

    // Validate if requested
    if validate {
        let y_expected = cpu_reference(
            n,
            m,
            k,
            &x_f32,
            &weights_packed,
            &scales.iter().map(|&v| v.to_f32()).collect::<Vec<_>>(),
            &biases.iter().map(|&v| v.to_f32()).collect::<Vec<_>>(),
        );

        let y_ptr = y_buf.contents() as *const f16;
        let y_out = unsafe { std::slice::from_raw_parts(y_ptr, m * n) };
        let y_out_f32: Vec<f32> = y_out.iter().map(|&v| v.to_f32()).collect();

        let tol = 1.0;
        let display_size = if n == 1 {
            m
        } else {
            n
        };
        println!(
            "first {} expected: {:?}",
            display_size,
            &y_expected[..display_size]
        );
        println!(
            "first {} got: {:?}",
            display_size,
            &y_out_f32[..display_size]
        );

        for (i, (&exp, &got)) in
            y_expected.iter().zip(y_out_f32.iter()).enumerate()
        {
            let diff = (exp - got).abs();
            assert!(
                diff <= tol,
                "M={} N={} K={} idx {} diff {} exp {} got {}",
                m,
                n,
                k,
                i,
                diff,
                exp,
                got
            );
        }
    }

    elapsed.as_secs_f64()
}

fn run_gemv_test(
    ctx: &MTLContext,
    m: usize,
    k: usize,
) {
    println!("--- Testing GEMV M={}, K={} ---", m, k);
    execute_quantized_matmul(ctx, m, 1, k, "qmv_f16_g64_b4", 1, true);
    println!("✅ GEMV M={}, K={} passed", m, k);
}

#[test]
fn test_quant_gemv() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping GEMV test");
            return;
        },
    };

    run_gemv_test(&ctx, 3, 64);
    run_gemv_test(&ctx, 1, 128);
    run_gemv_test(&ctx, 7, 256);
    run_gemv_test(&ctx, 13, 512);
    run_gemv_test(&ctx, 13, 511);
    run_gemv_test(&ctx, 8, 64);
    run_gemv_test(&ctx, 9, 64);
    run_gemv_test(&ctx, 25, 128);
    run_gemv_test(&ctx, 31, 512);
}

fn run_gemm_test(
    ctx: &MTLContext,
    m: usize,
    n: usize,
    k: usize,
) {
    println!("--- Testing GEMM M={}, N={}, K={} ---", m, n, k);
    execute_quantized_matmul(
        ctx,
        m,
        n,
        k,
        "qmm_transposed_f16_g64_b4",
        1,
        true,
    );
    println!("✅ GEMM M={}, N={}, K={} passed", m, n, k);
}

#[test]
fn test_quant_gemm() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping GEMM test");
            return;
        },
    };

    run_gemm_test(&ctx, 3, 5, 64);
    run_gemm_test(&ctx, 1, 1, 128);
    run_gemm_test(&ctx, 7, 13, 256);
    run_gemm_test(&ctx, 13, 7, 511);
    run_gemm_test(&ctx, 9, 9, 64);
    run_gemm_test(&ctx, 8, 9, 64);
    run_gemm_test(&ctx, 25, 17, 128);
    run_gemm_test(&ctx, 31, 27, 511);
}

fn benchmark_quantized_gemv(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    iterations: usize,
) -> f64 {
    execute_quantized_matmul(ctx, m, 1, k, "qmv_f16_g64_b4", iterations, false)
}

fn benchmark_quantized_gemm(
    ctx: &MTLContext,
    m: usize,
    n: usize,
    k: usize,
    iterations: usize,
) -> f64 {
    execute_quantized_matmul(
        ctx,
        m,
        n,
        k,
        "qmm_transposed_f16_g64_b4",
        iterations,
        false,
    )
}

#[test]
fn test_quantized_matmul_performance() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping performance test");
            return;
        },
    };

    println!("\n🚀 QUANTIZED MATMUL PERFORMANCE BENCHMARKS");
    println!("==========================================");

    // Test different sizes
    let test_configs = vec![
        // (M, N, K, iterations, description)
        (512, 1, 512, 100, "Small GEMV (512×512 × 512×1)"),
        (1024, 1, 1024, 50, "Medium GEMV (1024×1024 × 1024×1)"),
        (2048, 1, 2048, 20, "Large GEMV (2048×2048 × 2048×1)"),
        (4096, 1, 4096, 10, "XLarge GEMV (4096×4096 × 4096×1)"),
        (512, 16, 512, 50, "Small GEMM (512×512 × 512×16)"),
        (1024, 32, 1024, 20, "Medium GEMM (1024×1024 × 1024×32)"),
        (2048, 64, 2048, 10, "Large GEMM (2048×2048 × 2048×64)"),
        (4096, 128, 4096, 5, "XLarge GEMM (4096×4096 × 4096×128)"),
    ];

    for (m, n, k, iterations, description) in test_configs {
        let ops = 2.0 * (m as f64) * (n as f64) * (k as f64); // FLOPs

        if n == 1 {
            // GEMV
            let time = benchmark_quantized_gemv(&ctx, m, k, iterations);
            let avg_time = time / (iterations as f64);
            let throughput = (ops / avg_time) / 1e12; // TFLOPS

            println!("{}", description);
            println!(
                "  Time: {:.4}ms/iter, Throughput: {:.4} TFLOPS",
                avg_time * 1000.0,
                throughput
            );
        } else {
            // GEMM
            let time = benchmark_quantized_gemm(&ctx, m, n, k, iterations);
            let avg_time = time / (iterations as f64);
            let throughput = (ops / avg_time) / 1e12; // TFLOPS

            println!("{}", description);
            println!(
                "  Time: {:.4}ms/iter, Throughput: {:.4} TFLOPS",
                avg_time * 1000.0,
                throughput
            );
        }
    }
}
