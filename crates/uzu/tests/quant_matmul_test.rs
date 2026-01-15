use std::time::Instant;

use half::{bf16, f16};
use metal::{Buffer, Device, MTLResourceOptions};
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::quant_matmul::{
            QuantizationType, QuantizedMatmulArguments, QuantizedMatmulKernel,
        },
    },
    config::QuantizationMode,
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

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
    weights_transposed: bool,
    bits: usize,
) -> Vec<u8> {
    let mut weights_quant: Vec<u8> = Vec::with_capacity(output_dim * input_dim);

    if weights_transposed {
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
    } else {
        // Weights stored as [input_dim × output_dim]
        for _row in 0..input_dim {
            for col in 0..output_dim {
                let v = if bits == 4 {
                    ((col + 1) & 0x0F) as u8
                } else {
                    (col & 0xFF) as u8 // Allow 0..255 range
                };
                weights_quant.push(v);
            }
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
        zero_points.get(row_idx * stride + group_idx).copied().unwrap_or(0)
            as f32
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
    weights_transposed: bool,
    group_size: usize,
    dtype: DataType,
    bits: usize,
    quantization_type: QuantizationType,
    zero_points: &[u8],
    zero_points_stride: usize,
) -> Vec<f32> {
    let num_groups_k = (input_dim + group_size - 1) / group_size;
    let num_groups_n = (output_dim + group_size - 1) / group_size;
    let num_groups = if !weights_transposed {
        num_groups_n
    } else {
        num_groups_k
    };

    let mut y = vec![0.0f32; batch * output_dim];
    for i in 0..batch {
        for j in 0..output_dim {
            let mut acc = 0.0f32;
            if !weights_transposed {
                let group_idx = j / group_size;
                for l in 0..input_dim {
                    let weight_linear_idx = if weights_transposed {
                        j * input_dim + l
                    } else {
                        l * output_dim + j
                    };

                    let val_q = if bits == 4 {
                        get_4bit_value(b_quant, weight_linear_idx)
                    } else {
                        b_quant[weight_linear_idx] as f32
                    };

                    let val_a = a[i * input_dim + l];
                    let scale = scales[l * num_groups + group_idx];
                    let bias =
                        if quantization_type == QuantizationType::ZeroPoint {
                            let zp_val_qvm = get_zp_value(
                                zero_points,
                                zero_points_stride,
                                l,
                                group_idx,
                                bits,
                            );
                            -scale * zp_val_qvm
                        } else {
                            biases[l * num_groups + group_idx]
                        };
                    acc += val_a * (scale * val_q + bias);
                }
            } else {
                for g in 0..num_groups {
                    let scale = scales[j * num_groups + g];
                    let bias = match quantization_type {
                        QuantizationType::ZeroPoint => {
                            let zp = get_zp_value(
                                zero_points,
                                zero_points_stride,
                                j,
                                g,
                                bits,
                            );
                            -scale * zp
                        },
                        QuantizationType::Mlx => biases[j * num_groups + g],
                    };
                    let l_start = g * group_size;
                    let l_end = (l_start + group_size).min(input_dim);
                    let mut group_acc = 0.0f32;
                    let mut group_sum = 0.0f32;
                    for l in l_start..l_end {
                        let weight_linear_idx = if weights_transposed {
                            j * input_dim + l
                        } else {
                            l * output_dim + j
                        };

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
    quantization_type: QuantizationType,
    weights_transposed: bool,
    randomize_zp: bool,
) -> TestQuantParams {
    let num_groups_k = (input_dim + group_size - 1) / group_size;
    let num_groups_n = (output_dim + group_size - 1) / group_size;
    let num_groups = if !weights_transposed {
        num_groups_n
    } else {
        num_groups_k
    };
    let primary_dim = if !weights_transposed {
        input_dim
    } else {
        output_dim
    };

    let len = primary_dim * num_groups;
    let scales = vec![1.0; len];
    let scales_quant = quantize_slice(&scales, data_type);
    let mut biases = vec![0.0; len];

    let zero_points_stride = if bits == 4 {
        ((num_groups + 1) / 2).max(1)
    } else {
        num_groups
    };
    let zero_points_len = if !weights_transposed {
        input_dim * zero_points_stride
    } else {
        output_dim * zero_points_stride
    };
    let mut zero_points = vec![0u8; zero_points_len];

    if quantization_type == QuantizationType::ZeroPoint && randomize_zp {
        if !weights_transposed {
            // GMM/QVM non-transposed: Use [K][N_groups] layout for generation
            for k in 0..input_dim {
                for g in 0..num_groups {
                    let k_eff = k / group_size;
                    let base_val = ((k_eff * 5 + g * 7) & 0xFF) as u8;
                    let zp_val_u8 = if bits == 4 {
                        base_val & 0x0F
                    } else {
                        base_val
                    };
                    if bits == 4 {
                        let byte_index = k * zero_points_stride + (g >> 1);
                        if (g & 1) == 0 {
                            zero_points[byte_index] = (zero_points[byte_index]
                                & 0xF0)
                                | (zp_val_u8 & 0x0F);
                        } else {
                            zero_points[byte_index] = (zero_points[byte_index]
                                & 0x0F)
                                | ((zp_val_u8 & 0x0F) << 4);
                        }
                    } else {
                        zero_points[k * zero_points_stride + g] = zp_val_u8;
                    }

                    // Note: scales are [input_dim][num_groups] in this case?
                    // No, scales are 1.0.
                    // Biases logic needs to use same indexing!
                    // But biases is flat buffer.
                    // Index: k * num_groups + g.
                    let s = scales_quant[k * num_groups + g];
                    let zp_val = if bits == 4 {
                        (zp_val_u8 & 0x0F) as f32
                    } else {
                        zp_val_u8 as f32
                    };
                    biases[k * num_groups + g] =
                        quantize_value(-s * zp_val, data_type);
                }
            }
        } else {
            for j in 0..output_dim {
                for g in 0..num_groups {
                    let base_val = if weights_transposed {
                        j + 3 * g
                    } else {
                        3 * g
                    };
                    let zp_val: u8 = if bits == 4 {
                        (base_val as u8) & 0x0F
                    } else {
                        base_val as u8
                    };

                    if bits == 4 {
                        let byte_index = j * zero_points_stride + (g >> 1);
                        if (g & 1) == 0 {
                            zero_points[byte_index] = (zero_points[byte_index]
                                & 0xF0)
                                | (zp_val & 0x0F);
                        } else {
                            zero_points[byte_index] = (zero_points[byte_index]
                                & 0x0F)
                                | ((zp_val & 0x0F) << 4);
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
    }

    if quantization_type == QuantizationType::Mlx {
        if !weights_transposed {
            for k in 0..input_dim {
                for g in 0..num_groups {
                    let base_val = g * 3;
                    let bias_val = (base_val % 19) as f32 * 0.125;
                    biases[k * num_groups + g] =
                        quantize_value(bias_val, data_type);
                }
            }
        } else {
            for j in 0..output_dim {
                for g in 0..num_groups {
                    let base_val = if weights_transposed {
                        j * 7 + g * 3
                    } else {
                        g * 3
                    };
                    let bias_val = (base_val % 19) as f32 * 0.125;
                    biases[j * num_groups + g] =
                        quantize_value(bias_val, data_type);
                }
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
    batch: usize,
    input_dim: usize,
    output_dim: usize,
    weights_transposed: bool,
    iterations: usize,
    validate: bool,
    quantization_type: QuantizationType,
    randomize_zp: bool,
    group_size: usize,
    data_type: DataType,
    bits: usize,
) -> ExecutionResult {
    let weights_quant =
        create_test_weights(output_dim, input_dim, weights_transposed, bits);
    let weights_packed = if bits == 4 {
        pack_u4_weights(&weights_quant)
    } else {
        weights_quant.clone()
    };

    let params = generate_test_quant_params(
        output_dim,
        input_dim,
        group_size,
        data_type,
        bits,
        quantization_type,
        weights_transposed,
        randomize_zp,
    );
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

    let w_buf = ctx.device.new_buffer_with_data(
        weights_packed.as_ptr() as *const _,
        (weights_packed.len() * std::mem::size_of::<u8>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let s_buf = buffer_from_f32_slice(ctx, data_type, &params.scales);

    let b_buf = match quantization_type {
        QuantizationType::ZeroPoint => ctx.device.new_buffer_with_data(
            params.zero_points.as_ptr() as *const _,
            (params.zero_points.len() * std::mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ),
        QuantizationType::Mlx => {
            buffer_from_f32_slice(ctx, data_type, &params.biases)
        },
    };
    let x_buf = buffer_from_f32_slice(ctx, data_type, &x_f32);
    let y_buf = ctx.device.new_buffer(
        (batch * output_dim * data_type.size_in_bytes()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel = QuantizedMatmulKernel::new(
        &ctx,
        data_type,
        group_size,
        input_dim,
        output_dim,
        match bits {
            4 => QuantizationMode::UInt4,
            8 => QuantizationMode::Int8,
            _ => panic!("Unsupported bits: {}", bits),
        },
        quantization_type,
        weights_transposed,
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
                output_buffer: &y_buf,
                batch: batch as i32,
                input_dim: input_dim as i32,
                output_dim: output_dim as i32,
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

    let start = Instant::now();
    for _ in 0..iterations {
        let args = QuantizedMatmulArguments {
            a_buffer: &x_buf,
            a_offset: 0,
            b_buffer: &w_buf,
            scales_buffer: &s_buf,
            zero_points_or_biases_buffer: &b_buf,
            output_buffer: &y_buf,
            batch: batch as i32,
            input_dim: input_dim as i32,
            output_dim: output_dim as i32,
            quantization_type,
        };
        let cb_ref = ctx.command_queue.new_command_buffer();
        let encoder = cb_ref.new_compute_command_encoder();
        kernel.encode(encoder, args).unwrap();
        encoder.end_encoding();
        cb_ref.commit();
        cb_ref.wait_until_completed();
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
            weights_transposed,
            group_size,
            data_type,
            bits,
            quantization_type,
            &params.zero_points,
            params.zero_points_stride,
        );

        let y_out_f32: Vec<f32> = match data_type {
            DataType::F16 => {
                let y_ptr = y_buf.contents() as *const f16;
                let y_out = unsafe {
                    std::slice::from_raw_parts(y_ptr, batch * output_dim)
                };
                y_out.iter().map(|&v| v.to_f32()).collect()
            },
            DataType::BF16 => {
                let y_ptr = y_buf.contents() as *const bf16;
                let y_out = unsafe {
                    std::slice::from_raw_parts(y_ptr, batch * output_dim)
                };
                y_out.iter().map(|&v| v.to_f32()).collect()
            },
            DataType::F32 => {
                let y_ptr = y_buf.contents() as *const f32;
                let y_out = unsafe {
                    std::slice::from_raw_parts(y_ptr, batch * output_dim)
                };
                y_out.to_vec()
            },
            other => panic!("Unsupported dtype for validation: {:?}", other),
        };

        let mut debug_prints = 0;

        for (i, (&exp, &got)) in
            y_expected.iter().zip(y_out_f32.iter()).enumerate()
        {
            let diff = (exp - got).abs();

            if check_tolerance(exp, got) {
                if debug_prints < 16 {
                    println!(
                        "\n  detail idx {} diff {} exp {} got {}",
                        i, diff, exp, got
                    );
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
            println!(
                "\nTotal errors: {} out of {} outputs",
                debug_prints,
                batch * output_dim
            );
            println!("Error range: indices {}-{}", first_error, last_error);
            println!(
                "Dims batch={} output_dim={} input_dim={}",
                batch, output_dim, input_dim
            );
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
    quant_type: QuantizationType,
    bits: usize,
    data_type: DataType,
    group_size: usize,
}

const QMV_DIMS: &[(usize, usize)] = &[(128, 512), (512, 1024), (1024, 4096)];
const QVM_DIMS: &[(usize, usize)] = &[(128, 512), (512, 1024), (1024, 4096)];
const QMM_DIMS: &[(usize, usize, usize)] =
    &[(64, 64, 64), (512, 512, 1024), (128, 128, 256)];

fn run_kernel_test(
    ctx: &MTLContext,
    batch: usize,
    output_dim: usize,
    input_dim: usize,
    weights_transposed: bool,
    config: &TestConfig,
    validate: bool,
    iterations: usize,
) -> ExecutionResult {
    let randomize_zp = config.quant_type == QuantizationType::ZeroPoint;

    execute_quantized_matmul(
        ctx,
        batch,
        input_dim,
        output_dim,
        weights_transposed,
        iterations,
        validate,
        config.quant_type,
        randomize_zp,
        config.group_size,
        config.data_type,
        config.bits,
    )
}

#[test]
fn test_quant_gmv() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping QMV test");
            return;
        },
    };

    let configs = vec![
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
    ];

    for config in &configs {
        for &(output_dim, input_dim) in QMV_DIMS {
            run_kernel_test(
                &ctx, 1, output_dim, input_dim, true, config, true, 1,
            );
        }
    }
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

    let configs = vec![
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
    ];

    for config in &configs {
        for &(output_dim, input_dim) in QVM_DIMS {
            run_kernel_test(
                &ctx, 1, output_dim, input_dim, false, config, true, 1,
            );
        }
    }
}

#[test]
fn test_quant_gmm() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping QMM test");
            return;
        },
    };

    let configs = vec![
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
    ];

    for config in &configs {
        for &(batch, output_dim, input_dim) in QMM_DIMS {
            run_kernel_test(
                &ctx, batch, output_dim, input_dim, false, config, true, 1,
            );
        }
    }
}

#[test]
fn test_quant_gmm_transposed() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping QMM transposed test");
            return;
        },
    };

    let configs = vec![
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 4,
            data_type: DataType::F32,
            group_size: 64,
        },
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 8,
            data_type: DataType::F32,
            group_size: 64,
        },
    ];

    for config in &configs {
        for &(batch, output_dim, input_dim) in QMM_DIMS {
            run_kernel_test(
                &ctx, batch, output_dim, input_dim, true, config, true, 1,
            );
        }
    }
}

#[test]
#[ignore]
fn test_quant_matmul_perf() {
    let ctx = match create_test_context() {
        Some(c) => c,
        None => {
            println!("Metal not available — skipping Perf test");
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
            quant_type: QuantizationType::Mlx,
            bits: 4,
            data_type: DataType::BF16,
            group_size: 128,
        },
        // 8-bit Mlx BF16
        TestConfig {
            quant_type: QuantizationType::Mlx,
            bits: 8,
            data_type: DataType::BF16,
            group_size: 128,
        },
        // 4-bit ZP F16
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
            bits: 4,
            data_type: DataType::F16,
            group_size: 64,
        },
        // 8-bit ZP F16
        TestConfig {
            quant_type: QuantizationType::ZeroPoint,
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

    for config in &configs {
        for &(batch, output_dim, input_dim) in &shapes {
            let result = run_kernel_test(
                &ctx, batch, output_dim, input_dim, false, config, false, 20,
            );
            let avg = result.elapsed / 20.0 * 1000.0; // ms

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
