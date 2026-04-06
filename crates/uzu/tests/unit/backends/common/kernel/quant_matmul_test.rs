use std::time::Instant;

#[path = "../../../../common/helpers.rs"]
mod helpers;

use half::{bf16, f16};

use crate::{
    DataType,
    backends::common::{
        Backend, Buffer, Context, Encoder,
        gpu_types::QuantizationMode,
        kernel::quant_matmul::{
            QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernelEncodable,
            QuantizedMatmulType, tests::helpers::alloc_buffer_with_data,
        },
    },
    for_each_non_cpu_backend,
};

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

struct ExecutionResult {
    elapsed: f64,
}

fn create_test_weights(
    output_dim: usize,
    input_dim: usize,
    bits: usize,
) -> Vec<u8> {
    let mut weights_quant: Vec<u8> = Vec::with_capacity(output_dim * input_dim);

    // Weights stored as [output_dim × input_dim]
    for row in 0..output_dim {
        for _col in 0..input_dim {
            let v = if bits == 4 {
                ((row + 1) & 0x0F) as u8
            } else {
                ((row + 1) & 0xFF) as u8
            };
            weights_quant.push(v);
        }
    }

    weights_quant
}

fn check_tolerance(
    exp: f32,
    got: f32,
) -> bool {
    let diff = (exp - got).abs() as f64;
    // For F16 at boundaries (like 65504), machine epsilon is 32 (approx 0.05% relative error)
    // We use a conservative relative tolerance of 0.2%
    let rel_tol = 0.002;
    let abs_tol = 0.1f64;
    let tol = abs_tol.max(exp.abs() as f64 * rel_tol);
    diff > tol
}

fn get_4bit_value(
    data: &[u8],
    index: usize,
) -> f32 {
    let word_idx = index / 4;
    let word_offset = index % 4;
    let byte_idx = word_idx * 2;

    let word = if byte_idx + 1 < data.len() {
        data[byte_idx] as u16 | ((data[byte_idx + 1] as u16) << 8)
    } else {
        0
    };

    ((word >> (word_offset * 4)) & 0x0F) as f32
}

fn get_zp_value(
    zero_points: &[u8],
    stride: usize,
    row_idx: usize,
    group_idx: usize,
    bits: usize,
) -> f32 {
    if bits == 4 {
        let byte_index = row_idx * stride + (group_idx >> 1);
        let byte = zero_points.get(byte_index).copied().unwrap_or(0);
        if (group_idx & 1) == 0 {
            (byte & 0x0F) as f32
        } else {
            ((byte >> 4) & 0x0F) as f32
        }
    } else {
        zero_points.get(row_idx * stride + group_idx).copied().unwrap_or(0) as f32
    }
}

fn cpu_reference(
    batch: usize,
    output_dim: usize,
    input_dim: usize,
    a: &[f32],
    b_quant: &[u8],
    scales: &[f32],
    biases: &[f32],
    group_size: usize,
    dtype: DataType,
    bits: usize,
    quantization_type: QuantizedMatmulType,
    zero_points: &[u8],
    zero_points_stride: usize,
) -> Vec<f32> {
    let num_groups = (input_dim + group_size - 1) / group_size;

    let mut y = vec![0.0f32; batch * output_dim];
    for i in 0..batch {
        for j in 0..output_dim {
            let mut acc = 0.0f32;
            for g in 0..num_groups {
                let scale = scales[j * num_groups + g];
                let bias = match quantization_type {
                    QuantizedMatmulType::ZeroPoint => {
                        let zp = get_zp_value(zero_points, zero_points_stride, j, g, bits);
                        -scale * zp
                    },
                    QuantizedMatmulType::Mlx => biases[j * num_groups + g],
                };
                let l_start = g * group_size;
                let l_end = (l_start + group_size).min(input_dim);
                let mut group_acc = 0.0f32;
                let mut group_sum = 0.0f32;
                for l in l_start..l_end {
                    let weight_linear_idx = j * input_dim + l;

                    let val_q = if bits == 4 {
                        get_4bit_value(b_quant, weight_linear_idx)
                    } else {
                        b_quant[weight_linear_idx] as f32
                    };

                    let val_a = a[i * input_dim + l];
                    group_acc += val_a * val_q;
                    group_sum += val_a;
                }
                acc += scale * group_acc + bias * group_sum;
            }
            y[i * output_dim + j] = quantize_value(acc, dtype);
        }
    }
    y
}

struct TestQuantParams {
    scales: Vec<f32>,
    scales_quant: Vec<f32>,
    biases: Vec<f32>,
    zero_points: Vec<u8>,
    zero_points_stride: usize,
}

fn generate_test_quant_params(
    output_dim: usize,
    input_dim: usize,
    group_size: usize,
    data_type: DataType,
    bits: usize,
    quantization_type: QuantizedMatmulType,
    randomize_zp: bool,
) -> TestQuantParams {
    let num_groups = (input_dim + group_size - 1) / group_size;

    let len = output_dim * num_groups;
    let scales = vec![1.0; len];
    let scales_quant = quantize_slice(&scales, data_type);
    let mut biases = vec![0.0; len];

    let zero_points_stride = if bits == 4 {
        ((num_groups + 1) / 2).max(1)
    } else {
        num_groups
    };
    let mut zero_points = vec![0u8; output_dim * zero_points_stride];

    if quantization_type == QuantizedMatmulType::ZeroPoint && randomize_zp {
        for j in 0..output_dim {
            for g in 0..num_groups {
                let base_val = j + 3 * g;
                let zp_val: u8 = if bits == 4 {
                    (base_val as u8) & 0x0F
                } else {
                    base_val as u8
                };

                if bits == 4 {
                    let byte_index = j * zero_points_stride + (g >> 1);
                    if (g & 1) == 0 {
                        zero_points[byte_index] = (zero_points[byte_index] & 0xF0) | (zp_val & 0x0F);
                    } else {
                        zero_points[byte_index] = (zero_points[byte_index] & 0x0F) | ((zp_val & 0x0F) << 4);
                    }
                } else {
                    let byte_index = j * zero_points_stride + g;
                    zero_points[byte_index] = zp_val;
                }

                let s = scales_quant[j * num_groups + g];
                let b = quantize_value(-s * (zp_val as f32), data_type);
                biases[j * num_groups + g] = b;
            }
        }
    }

    if quantization_type == QuantizedMatmulType::Mlx {
        for j in 0..output_dim {
            for g in 0..num_groups {
                let base_val = j * 7 + g * 3;
                let bias_val = (base_val % 19) as f32 * 0.125;
                biases[j * num_groups + g] = quantize_value(bias_val, data_type);
            }
        }
    }

    TestQuantParams {
        scales,
        scales_quant,
        biases,
        zero_points,
        zero_points_stride,
    }
}

fn buffer_from_f32_slice<B: Backend>(
    ctx: &B::Context,
    dtype: DataType,
    values: &[f32],
) -> B::Buffer {
    match dtype {
        DataType::F16 => {
            let data: Vec<f16> = values.iter().map(|&v| f16::from_f32(v)).collect();
            alloc_buffer_with_data::<B, f16>(ctx, data.as_slice())
        },
        DataType::BF16 => {
            let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
            alloc_buffer_with_data::<B, bf16>(ctx, data.as_slice())
        },
        DataType::F32 => alloc_buffer_with_data::<B, f32>(ctx, values),
        other => {
            panic!("Unsupported dtype for buffer_from_f32_slice: {:?}", other)
        },
    }
}

fn execute_quantized_matmul<B: Backend>(
    ctx: &B::Context,
    batch: usize,
    input_dim: usize,
    output_dim: usize,
    iterations: usize,
    validate: bool,
    quantization_type: QuantizedMatmulType,
    randomize_zp: bool,
    group_size: usize,
    data_type: DataType,
    bits: usize,
) -> ExecutionResult {
    let weights_quant = create_test_weights(output_dim, input_dim, bits);
    let weights_packed = if bits == 4 {
        pack_u4_weights(&weights_quant)
    } else {
        weights_quant.clone()
    };

    let params =
        generate_test_quant_params(output_dim, input_dim, group_size, data_type, bits, quantization_type, randomize_zp);
    // X is batch × input_dim
    let x_f32: Vec<f32> = if batch == 1 {
        // Vector case: 1 × input_dim
        (1..=input_dim).map(|i| i as f32 / input_dim as f32).collect()
    } else {
        let mut x_vals: Vec<f32> = Vec::with_capacity(batch * input_dim);
        for _row in 0..batch {
            for i in 0..input_dim {
                x_vals.push((i + 1) as f32 / input_dim as f32);
            }
        }
        x_vals
    };
    let x_quant = quantize_slice(&x_f32, data_type);

    let w_buf = alloc_buffer_with_data::<B, u8>(ctx, &weights_packed);
    let s_buf = buffer_from_f32_slice::<B>(ctx, data_type, &params.scales);

    let b_buf = match quantization_type {
        QuantizedMatmulType::ZeroPoint => alloc_buffer_with_data::<B, u8>(ctx, &params.zero_points),
        QuantizedMatmulType::Mlx => buffer_from_f32_slice::<B>(ctx, data_type, &params.biases),
    };
    let x_buf = buffer_from_f32_slice::<B>(ctx, data_type, &x_f32);
    let mut y_buf = ctx.create_buffer(batch * output_dim * data_type.size_in_bytes()).expect("Failed to create buffer");

    let kernel = QuantizedMatmulKernelEncodable::<B>::new(
        &ctx,
        QuantizedMatmulConfiguration {
            data_type,
            group_size,
            input_dim,
            output_dim,
            mode: match bits {
                4 => QuantizationMode::UINT4,
                8 => QuantizationMode::INT8,
                _ => panic!("Unsupported bits: {}", bits),
            },
            quantization_type,
        },
    )
    .unwrap();

    if iterations > 1 {
        for _ in 0..3 {
            let args = QuantizedMatmulArguments {
                a_buffer: &x_buf,
                a_offset: 0,
                b_buffer: &w_buf,
                scales_buffer: &s_buf,
                zero_points_or_biases_buffer: &b_buf,
                output_buffer: &mut y_buf,
                batch_dim: batch,
            };
            let mut encoder = Encoder::new(ctx).unwrap();
            kernel.encode(&mut encoder, args).unwrap();
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let args = QuantizedMatmulArguments {
            a_buffer: &x_buf,
            a_offset: 0,
            b_buffer: &w_buf,
            scales_buffer: &s_buf,
            zero_points_or_biases_buffer: &b_buf,
            output_buffer: &mut y_buf,
            batch_dim: batch,
        };
        let mut encoder = Encoder::new(ctx).unwrap();
        kernel.encode(&mut encoder, args).unwrap();
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    }
    let elapsed = start.elapsed();

    if validate {
        let y_expected = cpu_reference(
            batch,
            output_dim,
            input_dim,
            &x_quant,
            &weights_packed,
            &params.scales_quant,
            &params.biases,
            group_size,
            data_type,
            bits,
            quantization_type,
            &params.zero_points,
            params.zero_points_stride,
        );

        let y_out_f32: Vec<f32> = match data_type {
            DataType::F16 => {
                let y_ptr = y_buf.cpu_ptr().as_ptr() as *const f16;
                let y_out = unsafe { std::slice::from_raw_parts(y_ptr, batch * output_dim) };
                y_out.iter().map(|&v| v.to_f32()).collect()
            },
            DataType::BF16 => {
                let y_ptr = y_buf.cpu_ptr().as_ptr() as *const bf16;
                let y_out = unsafe { std::slice::from_raw_parts(y_ptr, batch * output_dim) };
                y_out.iter().map(|&v| v.to_f32()).collect()
            },
            DataType::F32 => {
                let y_ptr = y_buf.cpu_ptr().as_ptr() as *const f32;
                let y_out = unsafe { std::slice::from_raw_parts(y_ptr, batch * output_dim) };
                y_out.to_vec()
            },
            other => panic!("Unsupported dtype for validation: {:?}", other),
        };

        let mut debug_prints = 0;

        for (i, (&exp, &got)) in y_expected.iter().zip(y_out_f32.iter()).enumerate() {
            let diff = (exp - got).abs();

            if check_tolerance(exp, got) {
                if debug_prints < 16 {
                    println!("\n  detail idx {} diff {} exp {} got {}", i, diff, exp, got);
                }
                debug_prints += 1;
            }
        }

        if debug_prints > 0 {
            let first_error = y_expected
                .iter()
                .zip(y_out_f32.iter())
                .enumerate()
                .find(|&(_, (&e, &g))| check_tolerance(e, g))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let last_error = y_expected
                .iter()
                .zip(y_out_f32.iter())
                .enumerate()
                .filter(|&(_, (&e, &g))| check_tolerance(e, g))
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            println!("\nTotal errors: {} out of {} outputs", debug_prints, batch * output_dim);
            println!("Error range: indices {}-{}", first_error, last_error);
            println!("Dims batch={} output_dim={} input_dim={}", batch, output_dim, input_dim);
            panic!("Validation failed with {} mismatches", debug_prints);
        }

        ExecutionResult {
            elapsed: elapsed.as_secs_f64(),
        }
    } else {
        ExecutionResult {
            elapsed: elapsed.as_secs_f64(),
        }
    }
}

struct TestConfig {
    quant_type: QuantizedMatmulType,
    bits: usize,
    data_type: DataType,
    group_size: usize,
}

const QMV_DIMS: &[(usize, usize)] = &[(128, 512), (512, 1024), (1024, 4096)];
const QMM_DIMS: &[(usize, usize, usize)] = &[(64, 64, 64), (512, 512, 1024), (128, 128, 256)];

fn run_kernel_test<B: Backend>(
    ctx: &B::Context,
    batch: usize,
    output_dim: usize,
    input_dim: usize,
    config: &TestConfig,
    validate: bool,
    iterations: usize,
) -> ExecutionResult {
    let randomize_zp = config.quant_type == QuantizedMatmulType::ZeroPoint;

    execute_quantized_matmul::<B>(
        ctx,
        batch,
        input_dim,
        output_dim,
        iterations,
        validate,
        config.quant_type,
        randomize_zp,
        config.group_size,
        config.data_type,
        config.bits,
    )
}

fn test_quant_gmv_internal<B: Backend>() {
    let ctx = match B::Context::new() {
        Ok(c) => c,
        Err(_) => {
            println!("{} backend not available — skipping QMV test", std::any::type_name::<B>());
            return;
        },
    };

    let configs = vec![
        TestConfig {
            quant_type: QuantizedMatmulType::ZeroPoint,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizedMatmulType::ZeroPoint,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizedMatmulType::Mlx,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizedMatmulType::Mlx,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
    ];

    for config in &configs {
        for &(output_dim, input_dim) in QMV_DIMS {
            run_kernel_test::<B>(&ctx, 1, output_dim, input_dim, config, true, 1);
        }
    }
}

fn test_quant_gmm_transposed_internal<B: Backend>() {
    let ctx = match B::Context::new() {
        Ok(c) => c,
        Err(_) => {
            println!("{} not available — skipping QMM transposed test", std::any::type_name::<B>());
            return;
        },
    };

    let configs = vec![
        TestConfig {
            quant_type: QuantizedMatmulType::ZeroPoint,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizedMatmulType::ZeroPoint,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizedMatmulType::Mlx,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizedMatmulType::Mlx,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
    ];

    for config in &configs {
        for &(batch, output_dim, input_dim) in QMM_DIMS {
            run_kernel_test::<B>(&ctx, batch, output_dim, input_dim, config, true, 1);
        }
    }
}

fn test_quant_matmul_perf_internal<B: Backend>() {
    let ctx = match B::Context::new() {
        Ok(c) => c,
        Err(_) => {
            println!("{} backend not available — skipping Perf test", std::any::type_name::<B>());
            return;
        },
    };

    // Llama 3 8B approximate shapes
    let shapes = vec![
        // Decoding (M=1)
        (1, 4096, 4096),
        (1, 14336, 4096),
        (1, 4096, 14336),
        // Prefill (Small batch M=128)
        (128, 4096, 4096),
    ];

    let configs = vec![
        // 4-bit Mlx BF16
        TestConfig {
            quant_type: QuantizedMatmulType::Mlx,
            bits: 4,
            data_type: DataType::BF16,
            group_size: 128,
        },
        // 8-bit Mlx BF16
        TestConfig {
            quant_type: QuantizedMatmulType::Mlx,
            bits: 8,
            data_type: DataType::BF16,
            group_size: 128,
        },
        // 4-bit ZP F16
        TestConfig {
            quant_type: QuantizedMatmulType::ZeroPoint,
            bits: 4,
            data_type: DataType::F16,
            group_size: 64,
        },
        // 8-bit ZP F16
        TestConfig {
            quant_type: QuantizedMatmulType::ZeroPoint,
            bits: 8,
            data_type: DataType::F16,
            group_size: 64,
        },
    ];

    println!(
        "{:<20} | {:<10} | {:<6} | {:<5} | {:<5} | {:<15}",
        "Kernel Config", "M x N x K", "Type", "Bits", "Group", "Avg Duration"
    );
    println!("{}", "-".repeat(85));

    let iterations = 20;
    for config in &configs {
        for &(batch, output_dim, input_dim) in &shapes {
            let result = run_kernel_test::<B>(&ctx, batch, output_dim, input_dim, config, false, iterations);
            let avg = result.elapsed / (iterations as f64) * 1000.0; // ms

            println!(
                "{:<20} | {:<10} | {:<6} | {:<5} | {:<5} | {:.4} ms",
                format!("{:?}", config.quant_type),
                format!("{}x{}x{}", batch, output_dim, input_dim),
                format!("{:?}", config.data_type),
                config.bits,
                config.group_size,
                avg
            );
        }
    }
}

#[test]
fn test_quant_gmv() {
    for_each_non_cpu_backend!(|B| {
        test_quant_gmv_internal::<B>();
    })
}

#[test]
fn test_quant_gmm_transposed() {
    for_each_non_cpu_backend!(|B| {
        test_quant_gmm_transposed_internal::<B>();
    })
}

#[test]
#[ignore]
fn test_quant_matmul_perf() {
    for_each_non_cpu_backend!(|B| {
        test_quant_matmul_perf_internal::<B>();
    })
}
