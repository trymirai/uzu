use std::time::Instant;

use half::{bf16, f16};
use metal::{Buffer, Device, MTLResourceOptions};
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::quant_matmul::{
            QuantizationType, QuantizedMatmulArguments, QuantizedMatmulKernel,
            select_qmm_kernel_name,
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
    let mut out = Vec::with_capacity((values.len() + 3) / 4 * 2);
    for chunk in values.chunks(4) {
        let w0 = (*chunk.get(0).unwrap_or(&0) as u16) & 0x0F;
        let w1 = ((*chunk.get(1).unwrap_or(&0) as u16) & 0x0F) << 4;
        let w2 = ((*chunk.get(2).unwrap_or(&0) as u16) & 0x0F) << 8;
        let w3 = ((*chunk.get(3).unwrap_or(&0) as u16) & 0x0F) << 12;

        let word: u16 = w0 | w1 | w2 | w3;
        out.push(word as u8);
        out.push((word >> 8) as u8);
    }
    out
}

fn quantize_value(
    value: f32,
    dtype: DataType,
) -> f32 {
    match dtype {
        DataType::F16 => f16::from_f32(value).to_f32(),
        DataType::BF16 => bf16::from_f32(value).to_f32(),
        _ => value,
    }
}

fn quantize_slice(
    values: &[f32],
    dtype: DataType,
) -> Vec<f32> {
    values.iter().map(|&v| quantize_value(v, dtype)).collect()
}

struct ErrorStats {
    mean_abs: f64,
    max_abs: f64,
    count: usize,
}

struct ExecutionResult {
    elapsed: f64,
    errors: Option<ErrorStats>,
}

fn cpu_reference(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],      // A matrix (M x K)
    b_quant: &[u8], // B matrix (N x K), quantized and packed
    scales: &[f32], // scales for B (N x ceil(K/group_size))
    biases: &[f32], // biases for B (N x ceil(K/group_size))
    transpose_b: bool,
    group_size: usize,
    dtype: DataType,
) -> Vec<f32> {
    let num_groups = (k + group_size - 1) / group_size;
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for g in 0..num_groups {
                let scale = scales[j * num_groups + g];
                let bias = biases[j * num_groups + g];
                let l_start = g * group_size;
                let l_end = (l_start + group_size).min(k);
                let mut group_acc = 0.0f32;
                let mut group_sum = 0.0f32;
                for l in l_start..l_end {
                    let weight_linear_idx = if transpose_b {
                        l * n + j
                    } else {
                        j * k + l
                    };

                    let word_idx = weight_linear_idx / 4;
                    let word_offset = weight_linear_idx % 4;
                    let byte_idx = word_idx * 2;

                    let word = if byte_idx + 1 < b_quant.len() {
                        b_quant[byte_idx] as u16
                            | ((b_quant[byte_idx + 1] as u16) << 8)
                    } else {
                        0
                    };

                    let val_q = match word_offset {
                        0 => word & 0x000F,
                        1 => (word & 0x00F0) >> 4,
                        2 => (word & 0x0F00) >> 8,
                        3 => (word & 0xF000) >> 12,
                        _ => 0,
                    } as f32;
                    let val_a = a[i * k + l];
                    group_acc += val_a * val_q;
                    group_sum += val_a;
                }
                acc += scale * group_acc + bias * group_sum;
            }
            y[i * n + j] = quantize_value(acc, dtype);
        }
    }
    y
}

fn buffer_from_f32_slice(
    ctx: &MTLContext,
    dtype: DataType,
    values: &[f32],
) -> Buffer {
    match dtype {
        DataType::F16 => {
            let data: Vec<f16> =
                values.iter().map(|&v| f16::from_f32(v)).collect();
            ctx.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        },
        DataType::BF16 => {
            let data: Vec<bf16> =
                values.iter().map(|&v| bf16::from_f32(v)).collect();
            ctx.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * std::mem::size_of::<bf16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        },
        DataType::F32 => ctx.device.new_buffer_with_data(
            values.as_ptr() as *const std::ffi::c_void,
            (values.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        other => {
            panic!("Unsupported dtype for buffer_from_f32_slice: {:?}", other)
        },
    }
}

fn execute_quantized_matmul(
    ctx: &MTLContext,
    m: usize,
    n: usize,
    k: usize,
    kernel_name: &str,
    cpu_transpose: bool,
    iterations: usize,
    validate: bool,
    strict_validation: bool,
    quantization_type: QuantizationType,
    randomize_zp: bool,
    group_size: usize,
    data_type: DataType,
) -> ExecutionResult {
    // Prepare deterministic weights: w(row,k) = row+1
    // Always create weights in the same layout (test_m × test_k)
    let mut weights_q4: Vec<u8> = Vec::with_capacity(m * k);
    for row in 0..m {
        for _k in 0..k {
            let v = (row as u8 + 1) & 0x0F;
            weights_q4.push(v);
        }
    }
    let weights_packed = pack_u4_weights(&weights_q4);

    let num_groups = (k + group_size - 1) / group_size;
    let scales_f32: Vec<f32> = vec![1.0; m * num_groups];
    let scales_quant = quantize_slice(&scales_f32, data_type);
    let mut biases_quant: Vec<f32> = vec![0.0; m * num_groups];

    let x_f32: Vec<f32> = if n == 1 {
        (1..=k).map(|i| i as f32 / k as f32).collect()
    } else {
        let mut x_vals: Vec<f32> = Vec::with_capacity(k * n);
        for _col in 0..n {
            for i in 0..k {
                x_vals.push((i + 1) as f32 / k as f32);
            }
        }
        x_vals
    };
    let x_quant = quantize_slice(&x_f32, data_type);

    let w_buf = ctx.device.new_buffer_with_data(
        weights_packed.as_ptr() as *const _,
        (weights_packed.len() * std::mem::size_of::<u8>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let s_buf = buffer_from_f32_slice(ctx, data_type, &scales_f32);

    let zero_points_stride = ((num_groups + 1) / 2).max(1);
    let mut zps_bytes = vec![0u8; m * zero_points_stride];
    if quantization_type == QuantizationType::ZeroPoint && randomize_zp {
        for j in 0..m {
            for g in 0..num_groups {
                let zp_val: u8 = ((j + 3 * g) as u8) & 0x0F;
                let byte_index = j * zero_points_stride + (g >> 1);
                if (g & 1) == 0 {
                    zps_bytes[byte_index] =
                        (zps_bytes[byte_index] & 0xF0) | (zp_val & 0x0F);
                } else {
                    zps_bytes[byte_index] =
                        (zps_bytes[byte_index] & 0x0F) | ((zp_val & 0x0F) << 4);
                }
                let s = scales_quant[j * num_groups + g];
                let b = quantize_value(-s * (zp_val as f32), data_type);
                biases_quant[j * num_groups + g] = b;
            }
        }
    }

    if quantization_type == QuantizationType::Mlx {
        for j in 0..m {
            for g in 0..num_groups {
                let bias_val = ((j * 7 + g * 3) % 19) as f32 * 0.125;
                biases_quant[j * num_groups + g] =
                    quantize_value(bias_val, data_type);
            }
        }
    }

    let b_buf = match quantization_type {
        QuantizationType::ZeroPoint => ctx.device.new_buffer_with_data(
            zps_bytes.as_ptr() as *const _,
            (zps_bytes.len() * std::mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        QuantizationType::Mlx => {
            buffer_from_f32_slice(ctx, data_type, &biases_quant)
        },
    };
    let x_buf = buffer_from_f32_slice(ctx, data_type, &x_f32);
    let y_buf = ctx.device.new_buffer(
        (m * n * data_type.size_in_bytes()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create kernel
    let kernel = QuantizedMatmulKernel::new(
        &ctx,
        data_type,
        kernel_name,
        quantization_type,
    )
    .unwrap();

    // Warmup (only for benchmarks)
    if iterations > 1 {
        for _ in 0..3 {
            let args = QuantizedMatmulArguments {
                a_buffer: &x_buf,
                b_buffer: &w_buf,
                scales_buffer: &s_buf,
                zero_points_or_biases_buffer: &b_buf,
                output_buffer: &y_buf,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                quantization_type,
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
            zero_points_or_biases_buffer: &b_buf,
            output_buffer: &y_buf,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            quantization_type: QuantizationType::ZeroPoint,
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
        let gpu_m = n; // GPU sees test_n as M
        let gpu_n = m; // GPU sees test_m as N  
        let gpu_k = k; // GPU sees test_k as K

        let y_expected = cpu_reference(
            gpu_m,
            gpu_n,
            gpu_k,
            &x_quant,
            &weights_packed,
            &scales_quant,
            &biases_quant,
            cpu_transpose,
            group_size,
            data_type,
        );

        let y_out_f32: Vec<f32> = match data_type {
            DataType::F16 => {
                let y_ptr = y_buf.contents() as *const f16;
                let y_out = unsafe { std::slice::from_raw_parts(y_ptr, m * n) };
                y_out.iter().map(|&v| v.to_f32()).collect()
            },
            DataType::BF16 => {
                let y_ptr = y_buf.contents() as *const bf16;
                let y_out = unsafe { std::slice::from_raw_parts(y_ptr, m * n) };
                y_out.iter().map(|&v| v.to_f32()).collect()
            },
            DataType::F32 => {
                let y_ptr = y_buf.contents() as *const f32;
                let y_out = unsafe { std::slice::from_raw_parts(y_ptr, m * n) };
                y_out.to_vec()
            },
            other => panic!("Unsupported dtype for validation: {:?}", other),
        };

        let tol = 1.0;
        let display_size = if n == 1 {
            32.min(m)
        } else {
            32.min(n)
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

        let mut total_error = 0.0f64;
        let mut max_error = 0.0f64;
        let mut count = 0usize;

        for (i, (&exp, &got)) in
            y_expected.iter().zip(y_out_f32.iter()).enumerate()
        {
            let diff = (exp - got).abs();
            total_error += diff as f64;
            max_error = max_error.max(diff as f64);
            count += 1;

            if diff > tol && strict_validation {
                panic!(
                    "M={} N={} K={} idx {} diff {} exp {} got {}",
                    m, n, k, i, diff, exp, got
                );
            }
        }

        ExecutionResult {
            elapsed: elapsed.as_secs_f64(),
            errors: Some(ErrorStats {
                mean_abs: total_error / count as f64,
                max_abs: max_error,
                count,
            }),
        }
    } else {
        ExecutionResult {
            elapsed: elapsed.as_secs_f64(),
            errors: None,
        }
    }
}

fn select_qmm_kernel_for_test(
    data_type: DataType,
    group_size: usize,
    transpose: bool,
    n: usize,
    k: usize,
) -> String {
    select_qmm_kernel_name(data_type, group_size, transpose, n, k)
        .expect("failed to resolve qmm kernel name")
}

fn run_gemv_test(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    group_size: usize,
    kernel_name: &str,
    data_type: DataType,
    quantization_type: QuantizationType,
) {
    println!("--- Testing GEMV M={}, K={} GS={} ---", m, k, group_size);
    let result = execute_quantized_matmul(
        ctx,
        m,
        1,
        k,
        kernel_name,
        false,
        1,
        true,
        true,
        quantization_type,
        false,
        group_size,
        data_type,
    );
    if let Some(stats) = result.errors {
        println!(
            "   Error: mean={:.4}, max={:.4}",
            stats.mean_abs, stats.max_abs
        );
    }
    println!("✅ GEMV M={}, K={} passed", m, k);
}

fn run_qvm_test(
    ctx: &MTLContext,
    n: usize,
    k: usize,
    group_size: usize,
    kernel_name: &str,
    data_type: DataType,
    quantization_type: QuantizationType,
) {
    println!("--- Testing QVM N={}, K={} GS={} ---", n, k, group_size);
    let _ = execute_quantized_matmul(
        ctx,
        1,
        n,
        k,
        kernel_name,
        false,
        1,
        true,
        true,
        quantization_type,
        false,
        group_size,
        data_type,
    );
    println!("✅ QVM N={}, K={} passed", n, k);
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

    run_gemv_test(
        &ctx,
        3,
        64,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        1,
        128,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        7,
        256,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        13,
        512,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        13,
        511,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        8,
        64,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        9,
        64,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        25,
        128,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        31,
        512,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
}

#[test]
fn test_quant_gemv_mlx_g128_bf16() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping MLX GEMV test");
            return;
        },
    };

    run_gemv_test(
        &ctx,
        128,
        128,
        128,
        "qmv_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::Mlx,
    );
    run_gemv_test(
        &ctx,
        256,
        256,
        128,
        "qmv_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::Mlx,
    );
    run_gemv_test(
        &ctx,
        4096,
        4096,
        128,
        "qmv_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::Mlx,
    );
}

#[test]
fn test_quant_gemv_fast_path() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping fast path test");
            return;
        },
    };

    println!("Testing fast path (N % 8 == 0 && K % 512 == 0)");

    run_gemv_test(
        &ctx,
        512,
        512,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::Mlx,
    );
    run_gemv_test(
        &ctx,
        1024,
        1024,
        128,
        "qmv_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::Mlx,
    );
    run_gemv_test(
        &ctx,
        4096,
        4096,
        128,
        "qmv_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::Mlx,
    );
    run_gemv_test(
        &ctx,
        2048,
        512,
        32,
        "qmv_f32_g32_b4",
        DataType::F32,
        QuantizationType::Mlx,
    );

    run_gemv_test(
        &ctx,
        512,
        512,
        64,
        "qmv_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        4096,
        4096,
        128,
        "qmv_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::ZeroPoint,
    );

    println!("✅ All fast path tests passed");
}

#[test]
fn test_quant_qvm() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping QVM test");
            return;
        },
    };

    run_qvm_test(
        &ctx,
        3,
        64,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_qvm_test(
        &ctx,
        1,
        128,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_qvm_test(
        &ctx,
        7,
        256,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_qvm_test(
        &ctx,
        13,
        512,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_qvm_test(
        &ctx,
        13,
        511,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_qvm_test(
        &ctx,
        8,
        64,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_qvm_test(
        &ctx,
        9,
        64,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_qvm_test(
        &ctx,
        25,
        128,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_qvm_test(
        &ctx,
        31,
        512,
        64,
        "qvm_f16_g64_b4",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
}

#[test]
fn test_quant_qvm_mlx_g128_bf16() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping MLX QVM test");
            return;
        },
    };

    run_qvm_test(
        &ctx,
        256,
        128,
        128,
        "qvm_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::Mlx,
    );
    run_qvm_test(
        &ctx,
        4096,
        4096,
        128,
        "qvm_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::Mlx,
    );
    run_qvm_test(
        &ctx,
        11008,
        4096,
        128,
        "qvm_bf16_g128_b4",
        DataType::BF16,
        QuantizationType::Mlx,
    );
}

fn run_gemm_test(
    ctx: &MTLContext,
    m: usize,
    n: usize,
    k: usize,
) {
    println!("--- Testing GEMM M={}, N={}, K={} ---", m, n, k);
    let kernel_name =
        select_qmm_kernel_for_test(DataType::F16, 64, false, n, k);
    let _ = execute_quantized_matmul(
        ctx,
        m,
        n,
        k,
        kernel_name.as_str(),
        true,
        1,
        true,
        true,
        QuantizationType::ZeroPoint,
        false,
        64,
        DataType::F16,
    );
    println!("✅ GEMM M={}, N={}, K={} passed", m, n, k);
}

fn run_gemm_transposed_test(
    ctx: &MTLContext,
    m: usize,
    n: usize,
    k: usize,
) {
    println!("--- Testing GEMM Transposed M={}, N={}, K={} ---", m, n, k);
    let kernel_name = select_qmm_kernel_for_test(DataType::F16, 64, true, n, k);
    let _ = execute_quantized_matmul(
        ctx,
        m,
        n,
        k,
        kernel_name.as_str(),
        false,
        1,
        true,
        true,
        QuantizationType::ZeroPoint,
        true,
        64,
        DataType::F16,
    );
    println!("✅ GEMM Transposed M={}, N={}, K={} passed", m, n, k);
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

    run_gemm_test(&ctx, 1, 1, 128);
    run_gemm_test(&ctx, 8, 16, 256);
    run_gemm_test(&ctx, 16, 8, 511);
    run_gemm_test(&ctx, 9, 9, 64);
    run_gemm_test(&ctx, 25, 17, 128);
}

#[test]
fn test_quant_gemm_transposed() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping GEMM transposed test");
            return;
        },
    };

    run_gemm_transposed_test(&ctx, 1, 1, 128);
    run_gemm_transposed_test(&ctx, 8, 16, 256);
    run_gemm_transposed_test(&ctx, 16, 8, 511);
    run_gemm_transposed_test(&ctx, 9, 9, 64);
    run_gemm_transposed_test(&ctx, 25, 17, 128);
}

#[test]
fn test_quant_f32_kernels() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping f32 kernel test");
            return;
        },
    };

    let _ = execute_quantized_matmul(
        &ctx,
        5,
        1,
        96,
        "qmv_f32_g32_b4",
        false,
        1,
        true,
        true,
        QuantizationType::ZeroPoint,
        true,
        32,
        DataType::F32,
    );

    let _ = execute_quantized_matmul(
        &ctx,
        1,
        5,
        96,
        "qvm_f32_g32_b4",
        false,
        1,
        true,
        true,
        QuantizationType::ZeroPoint,
        true,
        32,
        DataType::F32,
    );

    let kernel_name_qmm =
        select_qmm_kernel_for_test(DataType::F32, 32, false, 3, 96);
    let _ = execute_quantized_matmul(
        &ctx,
        4,
        3,
        96,
        kernel_name_qmm.as_str(),
        true,
        1,
        true,
        true,
        QuantizationType::ZeroPoint,
        false,
        32,
        DataType::F32,
    );

    let kernel_name_qmm_t =
        select_qmm_kernel_for_test(DataType::F32, 32, true, 4, 96);
    let _ = execute_quantized_matmul(
        &ctx,
        3,
        4,
        96,
        kernel_name_qmm_t.as_str(),
        false,
        1,
        true,
        true,
        QuantizationType::ZeroPoint,
        true,
        32,
        DataType::F32,
    );
}

fn benchmark_quantized_gemv(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    iterations: usize,
    group_size: usize,
    kernel_name: &str,
    data_type: DataType,
    quantization_type: QuantizationType,
) -> f64 {
    execute_quantized_matmul(
        ctx,
        m,
        1,
        k,
        kernel_name,
        false,
        iterations,
        false,
        false,
        quantization_type,
        false,
        group_size,
        data_type,
    )
    .elapsed
}

fn benchmark_quantized_qvm(
    ctx: &MTLContext,
    n: usize,
    k: usize,
    iterations: usize,
    group_size: usize,
    kernel_name: &str,
    data_type: DataType,
    quantization_type: QuantizationType,
) -> f64 {
    execute_quantized_matmul(
        ctx,
        1,
        n,
        k,
        kernel_name,
        false,
        iterations,
        false,
        false,
        quantization_type,
        false,
        group_size,
        data_type,
    )
    .elapsed
}

#[ignore]
#[test]
fn test_quantized_matmul_performance_mlx_g128_bf16() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping performance test");
            return;
        },
    };

    println!("\n🚀 MLX QUANTIZED GEMV/QVM PERFORMANCE (G128, BF16, 4-bit)");
    println!("============================================================");

    let test_configs = vec![
        // (M, N, K, iterations, description)
        // GEMV (N=1)
        (512, 1, 512, 100, "GEMV 512×512"),
        (1024, 1, 1024, 50, "GEMV 1024×1024"),
        (4096, 1, 4096, 200, "GEMV 4096×4096"),
        (11008, 1, 4096, 100, "GEMV 11008×4096 (Llama FFN)"),
        // QVM (M=1)
        (1, 512, 512, 100, "QVM 1×512 × 512×512"),
        (1, 4096, 4096, 200, "QVM 1×4096 × 4096×4096"),
        // GEMM Transposed (Batch > 1) - Simulating Prefill
        (4096, 128, 4096, 10, "GEMM Transposed 4096×128×4096"),
        (11008, 128, 4096, 10, "GEMM Transposed 11008×128×4096"),
    ];

    let quant_types = vec![
        (QuantizationType::Mlx, "MLX"),
        (QuantizationType::ZeroPoint, "ZeroPoint"),
    ];

    for (q_type, q_name) in quant_types {
        println!("\n--- {} Quantization ---", q_name);
        for (m, n, k, iterations, description) in &test_configs {
            let ops = 2.0 * (*m as f64) * (*n as f64) * (*k as f64);

            if *n == 1 {
                // GEMV
                let time = benchmark_quantized_gemv(
                    &ctx,
                    *m,
                    *k,
                    *iterations,
                    128,
                    "qmv_bf16_g128_b4_fast", // Force fast kernel
                    DataType::BF16,
                    q_type,
                );
                let avg_time = time / (*iterations as f64);
                let throughput = (ops / avg_time) / 1e12;
                println!(
                    "{}: {:.4}ms, {:.4} TFLOPS",
                    description,
                    avg_time * 1000.0,
                    throughput
                );
            } else if *m == 1 {
                // QVM
                let time = benchmark_quantized_qvm(
                    &ctx,
                    *n,
                    *k,
                    *iterations,
                    128,
                    "qvm_bf16_g128_b4",
                    DataType::BF16,
                    q_type,
                );
                let avg_time = time / (*iterations as f64);
                let throughput = (ops / avg_time) / 1e12;
                println!(
                    "{}: {:.4}ms, {:.4} TFLOPS",
                    description,
                    avg_time * 1000.0,
                    throughput
                );
            } else {
                // GEMM Transposed (assuming n > 1 and m > 1)
                let kernel_name = select_qmm_kernel_for_test(
                    DataType::BF16,
                    128,
                    true,
                    *n,
                    *k,
                );
                let result = execute_quantized_matmul(
                    &ctx,
                    *m,
                    *n,
                    *k,
                    &kernel_name,
                    false, // cpu_transpose (doesn't matter for perf)
                    *iterations,
                    false, // validate
                    false,
                    q_type,
                    false,
                    128,
                    DataType::BF16,
                );
                let avg_time = result.elapsed / (*iterations as f64);
                let throughput = (ops / avg_time) / 1e12;
                println!(
                    "{}: {:.4}ms, {:.4} TFLOPS",
                    description,
                    avg_time * 1000.0,
                    throughput
                );
            }
        }
    }
}

#[test]
fn test_quant_gemv_fast_kernels() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping fast kernel test");
            return;
        },
    };

    println!("Testing explicit fast kernels with ZeroPoint (AWQ)");
    run_gemv_test(
        &ctx,
        512,
        512,
        64,
        "qmv_f16_g64_b4_fast",
        DataType::F16,
        QuantizationType::ZeroPoint,
    );
    run_gemv_test(
        &ctx,
        1024,
        1024,
        128,
        "qmv_bf16_g128_b4_fast",
        DataType::BF16,
        QuantizationType::ZeroPoint,
    );

    println!("Testing explicit fast kernels with MLX");
    run_gemv_test(
        &ctx,
        512,
        512,
        64,
        "qmv_f16_g64_b4_fast",
        DataType::F16,
        QuantizationType::Mlx,
    );
}
