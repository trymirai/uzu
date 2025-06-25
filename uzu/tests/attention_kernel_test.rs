#![cfg(any(target_os = "macos", target_os = "ios"))]

mod common;

use std::mem::size_of;

use metal::{Device, MTLResourceOptions};
use ndarray::{Array2, Array3, Array4, s};
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::attention::{
        AttentionKernel, AttentionKernelVariant, AttentionSinglePassArguments,
        AttentionTwoPassArguments,
    },
    metal_extensions::command_buffer_extensions::CommandBufferTimingAccess,
};

fn reference_attention(
    queries: &Array4<f32>, // [batch, num_heads, seq_len, head_dim]
    keys: &Array4<f32>,    // [batch, num_kv_heads, seq_len, head_dim]
    values: &Array4<f32>,  // [batch, num_kv_heads, seq_len, head_dim]
    mask: Option<&Array3<f32>>, // [batch, seq_len, seq_len] or None
    scale: f32,
) -> Array4<f32> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_, num_kv_heads, kv_seq_len, _) = keys.dim();
    let n_repeats = num_heads / num_kv_heads;

    let scaled_queries = queries.mapv(|x| x * scale);

    let mut output = Array4::zeros((batch_size, num_heads, seq_len, head_dim));

    for b in 0..batch_size {
        if n_repeats > 1 {
            for kv_head in 0..num_kv_heads {
                for repeat in 0..n_repeats {
                    let q_head = kv_head * n_repeats + repeat;

                    let q = scaled_queries.slice(s![b, q_head, .., ..]); // [L, D]
                    let k = keys.slice(s![b, kv_head, .., ..]); // [L_kv, D]
                    let v = values.slice(s![b, kv_head, .., ..]); // [L_kv, D]

                    let mut scores = Array2::zeros((seq_len, kv_seq_len));
                    for i in 0..seq_len {
                        for j in 0..kv_seq_len {
                            let mut score = 0.0;
                            for d in 0..head_dim {
                                score += q[[i, d]] * k[[j, d]];
                            }
                            scores[[i, j]] = score;
                        }
                    }

                    if let Some(mask_data) = mask {
                        for i in 0..seq_len {
                            for j in 0..kv_seq_len.min(seq_len) {
                                scores[[i, j]] += mask_data[[b, i, j]];
                            }
                        }
                    }

                    for i in 0..seq_len {
                        let mut max_score = f32::NEG_INFINITY;
                        for j in 0..kv_seq_len {
                            max_score = max_score.max(scores[[i, j]]);
                        }

                        let mut sum_exp = 0.0;
                        for j in 0..kv_seq_len {
                            scores[[i, j]] = (scores[[i, j]] - max_score).exp();
                            sum_exp += scores[[i, j]];
                        }

                        for j in 0..kv_seq_len {
                            scores[[i, j]] /= sum_exp;
                        }
                    }

                    for i in 0..seq_len {
                        for d in 0..head_dim {
                            let mut out_val = 0.0;
                            for j in 0..kv_seq_len {
                                out_val += scores[[i, j]] * v[[j, d]];
                            }
                            output[[b, q_head, i, d]] = out_val;
                        }
                    }
                }
            }
        } else {
            for h in 0..num_heads {
                let q = scaled_queries.slice(s![b, h, .., ..]); // [L, D]
                let k = keys.slice(s![b, h, .., ..]); // [L_kv, D]
                let v = values.slice(s![b, h, .., ..]); // [L_kv, D]

                let mut scores = Array2::zeros((seq_len, kv_seq_len));
                for i in 0..seq_len {
                    for j in 0..kv_seq_len {
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += q[[i, d]] * k[[j, d]];
                        }
                        scores[[i, j]] = score;
                    }
                }

                if let Some(mask_data) = mask {
                    for i in 0..seq_len {
                        for j in 0..kv_seq_len.min(seq_len) {
                            scores[[i, j]] += mask_data[[b, i, j]];
                        }
                    }
                }

                for i in 0..seq_len {
                    let mut max_score = f32::NEG_INFINITY;
                    for j in 0..kv_seq_len {
                        max_score = max_score.max(scores[[i, j]]);
                    }

                    let mut sum_exp = 0.0;
                    for j in 0..kv_seq_len {
                        scores[[i, j]] = (scores[[i, j]] - max_score).exp();
                        sum_exp += scores[[i, j]];
                    }

                    for j in 0..kv_seq_len {
                        scores[[i, j]] /= sum_exp;
                    }
                }

                for i in 0..seq_len {
                    for d in 0..head_dim {
                        let mut out_val = 0.0;
                        for j in 0..kv_seq_len {
                            out_val += scores[[i, j]] * v[[j, d]];
                        }
                        output[[b, h, i, d]] = out_val;
                    }
                }
            }
        }
    }

    output
}

fn create_test_data(
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    seed: u64,
) -> (Array4<f32>, Array4<f32>, Array4<f32>, Array3<f32>) {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(seed);

    let queries = Array4::from_shape_fn(
        (batch_size, num_heads, seq_len, head_dim),
        |_| rng.random_range(-0.5..0.5),
    );

    let keys = Array4::from_shape_fn(
        (batch_size, num_kv_heads, seq_len, head_dim),
        |_| rng.random_range(-0.5..0.5),
    );

    let values = Array4::from_shape_fn(
        (batch_size, num_kv_heads, seq_len, head_dim),
        |_| rng.random_range(-0.5..0.5),
    );

    let mask =
        Array3::from_shape_fn((batch_size, seq_len, seq_len), |(_, i, j)| {
            if j <= i {
                0.0
            } else {
                -1e9
            }
        });

    (queries, keys, values, mask)
}

/// Convert ndarray to Metal buffer layout expected by our kernel
fn create_query_buffer(
    queries: &Array4<f32>,
    context: &MTLContext,
) -> metal::Buffer {
    let (_batch_size, num_heads, seq_len, head_dim) = queries.dim();

    // Our kernel expects queries layout: [num_heads, seq_len, head_dim]
    let mut query_data = vec![0.0f32; num_heads * seq_len * head_dim];

    for h in 0..num_heads {
        for t in 0..seq_len {
            for d in 0..head_dim {
                let idx = h * seq_len * head_dim + t * head_dim + d;
                query_data[idx] = queries[[0, h, t, d]];
            }
        }
    }

    context.device.new_buffer_with_data(
        query_data.as_ptr() as *const _,
        (query_data.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn create_key_cache_buffer(
    keys: &Array4<f32>,
    max_seq_len: usize,
    context: &MTLContext,
) -> metal::Buffer {
    let (_batch_size, num_kv_heads, seq_len, head_dim) = keys.dim();

    // Our kernel expects key cache layout: [num_kv_heads, max_seq_len, head_dim]
    let mut key_cache_data =
        vec![0.0f32; num_kv_heads * max_seq_len * head_dim];

    for h in 0..num_kv_heads {
        for t in 0..seq_len {
            for d in 0..head_dim {
                let idx = h * max_seq_len * head_dim + t * head_dim + d;
                key_cache_data[idx] = keys[[0, h, t, d]];
            }
        }
    }

    context.device.new_buffer_with_data(
        key_cache_data.as_ptr() as *const _,
        (key_cache_data.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn create_value_cache_buffer(
    values: &Array4<f32>,
    max_seq_len: usize,
    context: &MTLContext,
) -> metal::Buffer {
    let (_batch_size, num_kv_heads, seq_len, head_dim) = values.dim();

    // Our kernel expects value cache layout: [num_kv_heads, max_seq_len, head_dim]
    let mut value_cache_data =
        vec![0.0f32; num_kv_heads * max_seq_len * head_dim];

    for h in 0..num_kv_heads {
        for t in 0..seq_len {
            for d in 0..head_dim {
                let idx = h * max_seq_len * head_dim + t * head_dim + d;
                value_cache_data[idx] = values[[0, h, t, d]];
            }
        }
    }

    context.device.new_buffer_with_data(
        value_cache_data.as_ptr() as *const _,
        (value_cache_data.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn create_mask_buffer(
    mask: &Array3<f32>,
    num_heads: usize,
    context: &MTLContext,
) -> metal::Buffer {
    let (_batch_size, seq_len, _) = mask.dim();

    // Create mask buffer - the kernel expects mask layout: [num_heads, seq_len, seq_len]
    let mut mask_data = vec![0.0f32; num_heads * seq_len * seq_len];
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mask_idx = h * seq_len * seq_len + i * seq_len + j;
                mask_data[mask_idx] = mask[[0, i, j]];
            }
        }
    }

    context.device.new_buffer_with_data(
        mask_data.as_ptr() as *const _,
        (mask_data.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn convert_kernel_output(
    output_slice: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Array4<f32> {
    let mut kernel_output =
        Array4::zeros((batch_size, num_heads, seq_len, head_dim));
    for h in 0..num_heads {
        for t in 0..seq_len {
            for d in 0..head_dim {
                // Metal kernel output layout: o_offset = q_seq_idx * num_heads + head_idx
                let o_offset = t * num_heads + h;
                let idx = o_offset * head_dim + d;
                kernel_output[[0, h, t, d]] = output_slice[idx];
            }
        }
    }
    kernel_output
}

fn run_single_pass_attention(
    kernel: &AttentionKernel,
    context: &MTLContext,
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    mask: Option<&Array3<f32>>,
    scale: f32,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_batch_size, num_kv_heads, _seq_len, _head_dim) = keys.dim();

    let query_buffer = create_query_buffer(queries, context);
    let key_cache_buffer = create_key_cache_buffer(keys, seq_len, context);
    let value_cache_buffer =
        create_value_cache_buffer(values, seq_len, context);

    let mask_buffer = mask.map(|m| create_mask_buffer(m, num_heads, context));

    let output_buffer = context.device.new_buffer(
        (num_heads * seq_len * head_dim * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = context.command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();

    let args = AttentionSinglePassArguments {
        queries_buffer: &query_buffer,
        keys_buffer: &key_cache_buffer,
        values_buffer: &value_cache_buffer,
        output_buffer: &output_buffer,
        gqa_factor: (num_heads / num_kv_heads) as i32,
        sequence_length: seq_len as i32,
        k_head_stride: (seq_len * head_dim) as i32,
        k_seq_stride: head_dim as i32,
        v_head_stride: (seq_len * head_dim) as i32,
        v_seq_stride: head_dim as i32,
        scale,
        mask_buffer: mask_buffer.as_ref(),
        mask_kv_seq_stride: 1,
        mask_q_seq_stride: seq_len as i32,
        mask_head_stride: 0,
        window_size: seq_len as i32,
        ring_offset: 0,
        prefix_length: 0,
        num_heads,
        suffix_length: seq_len,
        head_dim,
    };

    kernel.encode_single_pass(&compute_encoder, args)?;

    compute_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output_ptr = output_buffer.contents() as *const f32;
    let output_slice = unsafe {
        std::slice::from_raw_parts(output_ptr, num_heads * seq_len * head_dim)
    };

    let kernel_output = convert_kernel_output(
        output_slice,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
    );

    Ok(kernel_output)
}

fn compare_results(
    kernel_output: &Array4<f32>,
    reference_output: &Array4<f32>,
    tolerance: f32,
    test_name: &str,
) -> Result<(), String> {
    let max_diff = kernel_output
        .iter()
        .zip(reference_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let relative_error = kernel_output
        .iter()
        .zip(reference_output.iter())
        .map(|(a, b)| {
            if b.abs() > 1e-6 {
                (a - b).abs() / b.abs()
            } else {
                (a - b).abs()
            }
        })
        .fold(0.0f32, f32::max);

    println!("Max absolute difference ({}): {}", test_name, max_diff);
    println!("Max relative error ({}): {}", test_name, relative_error);

    println!("Debug - Sample values ({}):", test_name);
    println!(
        "Kernel output [0,0,0,0:4]: {:?}",
        &kernel_output.slice(s![0, 0, 0, 0..4]).to_vec()
    );
    println!(
        "Reference output [0,0,0,0:4]: {:?}",
        &reference_output.slice(s![0, 0, 0, 0..4]).to_vec()
    );
    println!(
        "Kernel output [0,0,1,0:4]: {:?}",
        &kernel_output.slice(s![0, 0, 1, 0..4]).to_vec()
    );
    println!(
        "Reference output [0,0,1,0:4]: {:?}",
        &reference_output.slice(s![0, 0, 1, 0..4]).to_vec()
    );

    let kernel_max = kernel_output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let kernel_min =
        kernel_output.iter().fold(f32::INFINITY, |a, &b| a.min(b.abs()));
    let ref_max = reference_output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let ref_min =
        reference_output.iter().fold(f32::INFINITY, |a, &b| a.min(b.abs()));

    println!("Kernel output range: [{}, {}]", kernel_min, kernel_max);
    println!("Reference output range: [{}, {}]", ref_min, ref_max);

    if max_diff >= tolerance {
        return Err(format!(
            "{} output differs from reference by more than {}: max_diff = {}",
            test_name, tolerance, max_diff
        ));
    }

    println!("✓ {} test passed!", test_name);
    Ok(())
}

#[test]
fn test_single_pass_attention_basic() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("Failed to create MTLContext: {:?}", e);
        },
    };

    let batch_size = 1;
    let num_heads = 4;
    let num_kv_heads = 4;
    let seq_len = 8;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values, _mask) = create_test_data(
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        42,
    );

    println!("Testing reference implementation without mask...");
    let reference_output =
        reference_attention(&queries, &keys, &values, None, scale);

    let ref_max = reference_output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let ref_min =
        reference_output.iter().fold(f32::INFINITY, |a, &b| a.min(b.abs()));
    println!("Reference output range (no mask): [{}, {}]", ref_min, ref_max);
    println!(
        "Reference sample values: {:?}",
        &reference_output.slice(s![0, 0, 0, 0..4]).to_vec()
    );

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    let variant = kernel.choose_variant(seq_len, head_dim);
    println!("Using kernel variant: {:?}", variant);
    println!(
        "Supports single-pass for head_dim={}: {}",
        head_dim,
        kernel.supports_single_pass(head_dim)
    );
    println!(
        "Supports two-pass for head_dim={}: {}",
        head_dim,
        kernel.supports_two_pass(head_dim)
    );

    if !kernel.supports_single_pass(head_dim) {
        panic!("Single-pass kernel not supported for head_dim={}", head_dim);
    }

    let kernel_output = match run_single_pass_attention(
        &kernel, &context, &queries, &keys, &values, None, scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run single-pass attention: {:?}", e);
        },
    };

    let tolerance = 1e-1;
    if let Err(e) = compare_results(
        &kernel_output,
        &reference_output,
        tolerance,
        "Single-pass attention",
    ) {
        panic!("{}", e);
    }
}

#[test]
fn test_single_pass_attention_with_mask() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("Failed to create MTLContext: {:?}", e);
        },
    };

    let batch_size = 1;
    let num_heads = 4;
    let num_kv_heads = 4;
    let seq_len = 8;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values, mask) = create_test_data(
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        42,
    );

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    if !kernel.supports_single_pass(head_dim) {
        panic!("Single-pass kernel not supported for head_dim={}", head_dim);
    }

    let kernel_output = match run_single_pass_attention(
        &kernel,
        &context,
        &queries,
        &keys,
        &values,
        Some(&mask),
        scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run single-pass attention with mask: {:?}", e);
        },
    };

    let reference_output =
        reference_attention(&queries, &keys, &values, Some(&mask), scale);

    println!("Mask values:");
    for i in 0..seq_len.min(4) {
        for j in 0..seq_len.min(4) {
            print!("{:8.1} ", mask[[0, i, j]]);
        }
        println!();
    }

    let tolerance = 1e-2;
    if let Err(e) = compare_results(
        &kernel_output,
        &reference_output,
        tolerance,
        "Single-pass attention with mask",
    ) {
        panic!("{}", e);
    }
}

#[test]
fn test_single_pass_attention_gqa() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("Failed to create MTLContext: {:?}", e);
        },
    };

    let batch_size = 1;
    let num_heads = 8;
    let num_kv_heads = 2;
    let seq_len = 8;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values, _mask) = create_test_data(
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        42,
    );

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    let variant = kernel.choose_variant(seq_len, head_dim);
    println!("Using kernel variant for GQA: {:?}", variant);

    if !kernel.supports_single_pass(head_dim) {
        panic!("Single-pass kernel not supported for head_dim={}", head_dim);
    }

    let kernel_output = match run_single_pass_attention(
        &kernel, &context, &queries, &keys, &values, None, scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run single-pass attention GQA: {:?}", e);
        },
    };

    let reference_output =
        reference_attention(&queries, &keys, &values, None, scale);

    let tolerance = 1e-1;
    if let Err(e) = compare_results(
        &kernel_output,
        &reference_output,
        tolerance,
        "Single-pass attention GQA",
    ) {
        panic!("{}", e);
    }
}

fn run_two_pass_attention(
    kernel: &AttentionKernel,
    context: &MTLContext,
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    mask: Option<&Array3<f32>>,
    scale: f32,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_, num_kv_heads, _, _) = keys.dim();

    let queries_buffer = create_query_buffer(queries, context);
    let keys_buffer = create_key_cache_buffer(keys, seq_len, context);
    let values_buffer = create_value_cache_buffer(values, seq_len, context);
    let mask_buffer = mask.map(|m| create_mask_buffer(m, num_heads, context));

    let total_blocks_count = 32;
    let partials_size = num_heads * seq_len * total_blocks_count * head_dim;
    let sums_maxs_size = num_heads * seq_len * total_blocks_count;

    let partials_buffer = context.device.new_buffer(
        (partials_size * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let sums_buffer = context.device.new_buffer(
        (sums_maxs_size * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let maxs_buffer = context.device.new_buffer(
        (sums_maxs_size * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = context.device.new_buffer(
        (num_heads * seq_len * head_dim * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = context.command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();

    let args = AttentionTwoPassArguments {
        queries_buffer: &queries_buffer,
        keys_buffer: &keys_buffer,
        values_buffer: &values_buffer,
        partials_buffer: &partials_buffer,
        sums_buffer: &sums_buffer,
        maxs_buffer: &maxs_buffer,
        output_buffer: &output_buffer,
        gqa_factor: (num_heads / num_kv_heads) as i32,
        sequence_length: seq_len as i32,
        k_head_stride: (seq_len * head_dim) as i32,
        k_seq_stride: head_dim as i32,
        v_head_stride: (seq_len * head_dim) as i32,
        v_seq_stride: head_dim as i32,
        scale,
        mask_buffer: mask_buffer.as_ref(),
        mask_kv_seq_stride: 1,
        mask_q_seq_stride: seq_len as i32,
        mask_head_stride: 0,
        window_size: seq_len as i32,
        ring_offset: 0,
        prefix_length: 0,
        num_heads,
        suffix_length: seq_len,
        head_dim,
    };

    kernel.encode_two_pass(&compute_encoder, args)?;

    compute_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output_ptr = output_buffer.contents() as *const f32;
    let output_slice = unsafe {
        std::slice::from_raw_parts(output_ptr, num_heads * seq_len * head_dim)
    };

    let kernel_output = convert_kernel_output(
        output_slice,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
    );

    Ok(kernel_output)
}

#[test]
fn test_two_pass_attention() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("Failed to create MTLContext: {:?}", e);
        },
    };

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    let batch_size = 1;
    let num_heads = 8;
    let num_kv_heads: usize = 8;
    let seq_len = 2048;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    if !kernel.supports_two_pass(head_dim) {
        panic!("Two-pass kernel not supported for head_dim={}", head_dim);
    }

    let variant = kernel.choose_variant(seq_len, head_dim);
    if !matches!(variant, AttentionKernelVariant::TwoPass) {
        panic!(
            "Two-pass not selected for seq_len={}. Got {:?}",
            seq_len, variant
        );
    }

    let (queries, keys, values, _mask) = create_test_data(
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        42,
    );

    let reference_output =
        reference_attention(&queries, &keys, &values, None, scale);

    let kernel_output = match run_two_pass_attention(
        &kernel, &context, &queries, &keys, &values, None, scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run two-pass attention: {:?}", e);
        },
    };

    let tolerance = 1e-1;
    if let Err(e) = compare_results(
        &kernel_output,
        &reference_output,
        tolerance,
        "Two-pass attention",
    ) {
        panic!("{}", e);
    }
}

#[test]
fn test_two_pass_attention_gqa() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("Failed to create MTLContext: {:?}", e);
        },
    };

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    let batch_size = 1;
    let num_heads = 8;
    let num_kv_heads = 2;
    let seq_len = 4096;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    if !kernel.supports_two_pass(head_dim) {
        panic!("Two-pass kernel not supported for head_dim={}", head_dim);
    }

    let variant = kernel.choose_variant(seq_len, head_dim);
    if !matches!(variant, AttentionKernelVariant::TwoPass) {
        panic!(
            "Two-pass not selected for GQA seq_len={}. Got {:?}",
            seq_len, variant
        );
    }

    let (queries, keys, values, _mask) = create_test_data(
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        42,
    );

    let reference_output =
        reference_attention(&queries, &keys, &values, None, scale);

    let kernel_output = match run_two_pass_attention(
        &kernel, &context, &queries, &keys, &values, None, scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run two-pass attention GQA: {:?}", e);
        },
    };

    let tolerance = 1e-1;
    if let Err(e) = compare_results(
        &kernel_output,
        &reference_output,
        tolerance,
        "Two-pass attention GQA",
    ) {
        panic!("{}", e);
    }
}

#[test]
fn perf_two_pass_attention() {
    use std::time::Instant;

    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("Failed to create MTLContext: {:?}", e);
        },
    };

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    // ---- Problem sizes requiring two-pass ----
    let batch_size = 1;
    let num_heads = 32;
    let num_kv_heads = 32;
    let seq_len = 8192; // Large sequence length (prefix + suffix)
    let suffix_length = 1; // Only processing 1 new token (realistic inference)
    let head_dim = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();

    if !kernel.supports_two_pass(head_dim) {
        println!(
            "Skipping two-pass perf test: not supported for head_dim={}",
            head_dim
        );
        return;
    }

    let variant = kernel.choose_variant(seq_len, head_dim);
    if !matches!(variant, AttentionKernelVariant::TwoPass) {
        println!(
            "Skipping two-pass perf test: variant {:?} selected instead",
            variant
        );
        return;
    }

    println!(
        "Creating test data for two-pass performance test (prefix={}, suffix={})...",
        seq_len - suffix_length,
        suffix_length
    );
    let (queries, keys, values, _mask) = create_test_data(
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        123,
    );

    // ---- Create buffers ----
    // For realistic inference, we only process queries for the suffix (new tokens)
    let queries_suffix =
        queries.slice(s![.., .., (seq_len - suffix_length).., ..]).to_owned();
    let queries_buffer = create_query_buffer(&queries_suffix, &context);
    let keys_buffer = create_key_cache_buffer(&keys, seq_len, &context);
    let values_buffer = create_value_cache_buffer(&values, seq_len, &context);

    let total_blocks_count = 32;
    let partials_size =
        num_heads * suffix_length * total_blocks_count * head_dim;
    let sums_maxs_size = num_heads * suffix_length * total_blocks_count;

    let partials_buffer = context.device.new_buffer(
        (partials_size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sums_buffer = context.device.new_buffer(
        (sums_maxs_size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let maxs_buffer = context.device.new_buffer(
        (sums_maxs_size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = context.device.new_buffer(
        (num_heads * suffix_length * head_dim * std::mem::size_of::<f32>())
            as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // ---- Launch and time ----
    let command_buffer = context.command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();

    let args = AttentionTwoPassArguments {
        queries_buffer: &queries_buffer,
        keys_buffer: &keys_buffer,
        values_buffer: &values_buffer,
        partials_buffer: &partials_buffer,
        sums_buffer: &sums_buffer,
        maxs_buffer: &maxs_buffer,
        output_buffer: &output_buffer,
        gqa_factor: (num_heads / num_kv_heads) as i32,
        sequence_length: seq_len as i32,
        k_head_stride: (seq_len * head_dim) as i32,
        k_seq_stride: head_dim as i32,
        v_head_stride: (seq_len * head_dim) as i32,
        v_seq_stride: head_dim as i32,
        scale,
        mask_buffer: None,
        mask_kv_seq_stride: 1,
        mask_q_seq_stride: suffix_length as i32,
        mask_head_stride: 0,
        window_size: seq_len as i32,
        ring_offset: 0,
        prefix_length: (seq_len - suffix_length) as i32,
        num_heads,
        suffix_length, // Use actual suffix_length, not seq_len
        head_dim,
    };

    kernel.encode_two_pass(&compute_encoder, args).expect("encode");
    compute_encoder.end_encoding();

    // Time both host-side and GPU execution
    let host_timer = Instant::now();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    // Get actual GPU execution time
    let gpu_elapsed_ms = command_buffer.gpu_execution_time_ms();

    match gpu_elapsed_ms {
        Some(gpu_time) => {
            println!(
                "Two-pass attention perf (heads={}, prefix={}, suffix={}, head_dim={}): GPU={:.2} ms, Host-side={:.2} ms",
                num_heads,
                seq_len - suffix_length,
                suffix_length,
                head_dim,
                gpu_time,
                host_elapsed_ms
            );
        },
        None => {
            println!(
                "Two-pass attention perf (heads={}, prefix={}, suffix={}, head_dim={}): Host-side={:.2} ms (GPU timing unavailable)",
                num_heads,
                seq_len - suffix_length,
                suffix_length,
                head_dim,
                host_elapsed_ms
            );
        },
    }

    // ---- Sanity check ----
    let output_ptr = output_buffer.contents() as *const f32;
    let output_slice = unsafe {
        std::slice::from_raw_parts(
            output_ptr,
            num_heads * suffix_length * head_dim,
        )
    };

    // Check for NaN/Inf
    for &val in output_slice.iter().take(100) {
        assert!(val.is_finite(), "Output contains non-finite values");
    }

    println!("✓ Two-pass attention performance test completed");
}

#[test]
fn test_kv_cache_update_ring_buffer_addressing() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("Failed to create MTLContext: {:?}", e);
        },
    };

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    println!("=== Testing KV Cache Update Ring Buffer Addressing ===");

    // Test parameters
    let num_groups = 2;
    let num_heads = 4; // Must be >= num_groups for GQA
    let window_size = 4;
    let head_dim = 8;
    let suffix_length = 2; // Test with multiple tokens

    // Test realistic sliding window scenarios
    let test_cases = vec![
        // (total_prefix_length, ring_offset, description)
        (2, 0, "Still filling window"),
        (4, 0, "Window just filled"),
        (5, 1, "One token past window"),
        (6, 2, "Two tokens past window"),
        (8, 0, "Multiple wraps, ring_offset=0"),
        (9, 1, "Multiple wraps, ring_offset=1"),
    ];

    for (total_prefix_length, expected_ring_offset, description) in test_cases {
        println!(
            "\n--- Testing: {} (prefix_length={}, ring_offset={}) ---",
            description, total_prefix_length, expected_ring_offset
        );

        // Create rotated keys with recognizable pattern: group*1000 + token*100 + dim
        let mut rotated_keys_data =
            vec![0.0f32; num_groups * suffix_length * head_dim];
        for g in 0..num_groups {
            for t in 0..suffix_length {
                for d in 0..head_dim {
                    let idx = g * suffix_length * head_dim + t * head_dim + d;
                    rotated_keys_data[idx] = (g * 1000 + t * 100 + d) as f32;
                }
            }
        }

        // Create QKV data with recognizable pattern for values: group*2000 + token*200 + dim
        let qkv_stride = num_heads * head_dim + 2 * num_groups * head_dim;
        let mut qkv_data = vec![0.0f32; suffix_length * qkv_stride];

        // Fill values section of QKV (starts after queries and keys)
        let value_offset_in_qkv = num_heads * head_dim + num_groups * head_dim;
        for t in 0..suffix_length {
            for g in 0..num_groups {
                for d in 0..head_dim {
                    let qkv_idx =
                        t * qkv_stride + value_offset_in_qkv + g * head_dim + d;
                    qkv_data[qkv_idx] = (g * 2000 + t * 200 + d) as f32;
                }
            }
        }

        // Initialize cache buffers with sentinel values (-999.0) to detect overwrites
        let key_cache_data =
            vec![-999.0f32; num_groups * window_size * head_dim];
        let value_cache_data =
            vec![-999.0f32; num_groups * window_size * head_dim];

        // Create Metal buffers
        let rotated_keys_buffer = context.device.new_buffer_with_data(
            rotated_keys_data.as_ptr() as *const _,
            (rotated_keys_data.len() * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let qkv_buffer = context.device.new_buffer_with_data(
            qkv_data.as_ptr() as *const _,
            (qkv_data.len() * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let key_cache_buffer = context.device.new_buffer_with_data(
            key_cache_data.as_ptr() as *const _,
            (key_cache_data.len() * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let value_cache_buffer = context.device.new_buffer_with_data(
            value_cache_data.as_ptr() as *const _,
            (value_cache_data.len() * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Execute KV cache update with REAL prefix_length
        let command_buffer = context.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        let args =
            uzu::backends::metal::kernel::attention::KVCacheUpdateArguments {
                rotated_keys_buffer: &rotated_keys_buffer,
                qkv_buffer: &qkv_buffer,
                key_cache_buffer: &key_cache_buffer,
                value_cache_buffer: &value_cache_buffer,
                num_groups,
                num_heads,
                head_dim,
                suffix_length,
                prefix_length: total_prefix_length, // Use REAL prefix length
                max_sequence_length: window_size,
                ring_offset: expected_ring_offset,
            };

        match kernel.encode_kv_cache_update(&compute_encoder, args) {
            Ok(_) => {
                compute_encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                // Read back the cache results
                let key_cache_ptr = key_cache_buffer.contents() as *const f32;
                let key_cache_result = unsafe {
                    std::slice::from_raw_parts(
                        key_cache_ptr,
                        num_groups * window_size * head_dim,
                    )
                };

                let value_cache_ptr =
                    value_cache_buffer.contents() as *const f32;
                let value_cache_result = unsafe {
                    std::slice::from_raw_parts(
                        value_cache_ptr,
                        num_groups * window_size * head_dim,
                    )
                };

                // Calculate expected write positions based on our Metal kernel logic
                for token_idx in 0..suffix_length {
                    let expected_write_pos =
                        if total_prefix_length < window_size {
                            // Still filling the window - use linear addressing
                            total_prefix_length + token_idx
                        } else {
                            // Window is full - use ring buffer addressing
                            (expected_ring_offset + token_idx) % window_size
                        };

                    println!(
                        "Token {}: expected write position = {} (prefix_length={}, ring_offset={})",
                        token_idx,
                        expected_write_pos,
                        total_prefix_length,
                        expected_ring_offset
                    );

                    // Verify keys and values were written correctly
                    for g in 0..num_groups {
                        for d in 0..head_dim {
                            let cache_idx = g * window_size * head_dim
                                + expected_write_pos * head_dim
                                + d;
                            let expected_key_value =
                                (g * 1000 + token_idx * 100 + d) as f32;
                            let actual_key_value = key_cache_result[cache_idx];

                            assert_eq!(
                                actual_key_value,
                                expected_key_value,
                                "Key mismatch at group={}, token={}, dim={}, case='{}': expected {}, got {}",
                                g,
                                token_idx,
                                d,
                                description,
                                expected_key_value,
                                actual_key_value
                            );

                            let expected_value_value =
                                (g * 2000 + token_idx * 200 + d) as f32;
                            let actual_value_value =
                                value_cache_result[cache_idx];

                            assert_eq!(
                                actual_value_value,
                                expected_value_value,
                                "Value mismatch at group={}, token={}, dim={}, case='{}': expected {}, got {}",
                                g,
                                token_idx,
                                d,
                                description,
                                expected_value_value,
                                actual_value_value
                            );
                        }
                    }
                }

                println!(
                    "✓ Ring buffer addressing verified for case: {}",
                    description
                );
            },
            Err(e) => {
                panic!(
                    "Failed to encode KV cache update for case '{}': {:?}",
                    description, e
                );
            },
        }
    }

    println!(
        "\n✓ KV Cache Update ring buffer addressing test completed successfully!"
    );
}

#[test]
fn test_sliding_window_attention() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("Failed to create MTLContext: {:?}", e);
        },
    };

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    println!("=== Testing Sliding Window Attention ===");

    // Test parameters - simulate a realistic sliding window scenario
    let batch_size = 1;
    let num_heads = 4;
    let num_kv_heads = 4;
    let window_size = 8;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Simulate processing new tokens with a sliding window
    let total_sequence_length = 12; // Total logical sequence length
    let prefix_length = 10; // Already processed tokens
    let suffix_length = 2; // New tokens to process
    let ring_offset = 2; // Ring buffer has wrapped around

    // Create test data for the full logical sequence
    let (full_queries, full_keys, full_values, _) = create_test_data(
        batch_size,
        num_heads,
        num_kv_heads,
        total_sequence_length,
        head_dim,
        42,
    );

    // Extract only the suffix queries (new tokens)
    let suffix_queries =
        full_queries.slice(s![.., .., prefix_length.., ..]).to_owned();

    // Create a ring buffer cache with the current window state
    // The cache contains the last `window_size` tokens in ring buffer order
    let mut ring_keys =
        Array4::zeros((batch_size, num_kv_heads, window_size, head_dim));
    let mut ring_values =
        Array4::zeros((batch_size, num_kv_heads, window_size, head_dim));

    // Fill the ring buffer with the appropriate tokens
    // Ring buffer positions 0..ring_offset contain the newest tokens
    // Ring buffer positions ring_offset..window_size contain older tokens
    for logical_pos in 0..window_size {
        let source_pos = prefix_length - window_size + logical_pos;
        let ring_pos = (ring_offset + logical_pos) % window_size;

        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                ring_keys[[0, h, ring_pos, d]] =
                    full_keys[[0, h, source_pos, d]];
                ring_values[[0, h, ring_pos, d]] =
                    full_values[[0, h, source_pos, d]];
            }
        }
    }

    // Create Metal buffers
    let queries_buffer = create_query_buffer(&suffix_queries, &context);
    let keys_buffer =
        create_key_cache_buffer(&ring_keys, window_size, &context);
    let values_buffer =
        create_value_cache_buffer(&ring_values, window_size, &context);

    let output_buffer = context.device.new_buffer(
        (num_heads * suffix_length * head_dim * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Run sliding window attention
    let command_buffer = context.command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();

    let args = AttentionSinglePassArguments {
        queries_buffer: &queries_buffer,
        keys_buffer: &keys_buffer,
        values_buffer: &values_buffer,
        output_buffer: &output_buffer,
        gqa_factor: (num_heads / num_kv_heads) as i32,
        sequence_length: window_size as i32, // Attention over window_size tokens
        k_head_stride: (window_size * head_dim) as i32,
        k_seq_stride: head_dim as i32,
        v_head_stride: (window_size * head_dim) as i32,
        v_seq_stride: head_dim as i32,
        scale,
        mask_buffer: None,
        mask_kv_seq_stride: 1,
        mask_q_seq_stride: window_size as i32,
        mask_head_stride: 0,
        window_size: window_size as i32,
        ring_offset: ring_offset as i32,
        prefix_length: prefix_length as i32,
        num_heads,
        suffix_length,
        head_dim,
    };

    match kernel.encode_single_pass(&compute_encoder, args) {
        Ok(_) => {
            compute_encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read back results
            let output_ptr = output_buffer.contents() as *const f32;
            let output_slice = unsafe {
                std::slice::from_raw_parts(
                    output_ptr,
                    num_heads * suffix_length * head_dim,
                )
            };

            // Convert to ndarray format
            let kernel_output = convert_kernel_output(
                output_slice,
                batch_size,
                num_heads,
                suffix_length,
                head_dim,
            );

            // Compute reference: attention between suffix queries and the window of keys/values
            // Create the reference input by extracting the window in logical order
            let window_keys = Array4::from_shape_fn(
                (batch_size, num_kv_heads, window_size, head_dim),
                |(b, h, t, d)| {
                    let source_pos = prefix_length - window_size + t;
                    full_keys[[b, h, source_pos, d]]
                },
            );

            let window_values = Array4::from_shape_fn(
                (batch_size, num_kv_heads, window_size, head_dim),
                |(b, h, t, d)| {
                    let source_pos = prefix_length - window_size + t;
                    full_values[[b, h, source_pos, d]]
                },
            );

            let reference_output = reference_attention(
                &suffix_queries,
                &window_keys,
                &window_values,
                None,
                scale,
            );

            // Compare results
            let tolerance = 1e-2;
            match compare_results(
                &kernel_output,
                &reference_output,
                tolerance,
                "Sliding window attention",
            ) {
                Ok(_) => {
                    println!("✓ Sliding window attention test passed!");

                    // Additional verification: check that values are reasonable
                    let output_max = kernel_output
                        .iter()
                        .fold(0.0f32, |a, &b| a.max(b.abs()));
                    let output_min = kernel_output
                        .iter()
                        .fold(f32::INFINITY, |a, &b| a.min(b.abs()));
                    println!(
                        "Sliding window output range: [{}, {}]",
                        output_min, output_max
                    );

                    // Verify no NaN/Inf values
                    for &val in kernel_output.iter() {
                        assert!(
                            val.is_finite(),
                            "Sliding window output contains non-finite values"
                        );
                    }

                    println!(
                        "✓ Sliding window attention produces finite, reasonable outputs"
                    );
                },
                Err(e) => {
                    panic!("Sliding window attention test failed: {}", e);
                },
            }
        },
        Err(e) => {
            panic!("Failed to encode sliding window attention: {:?}", e);
        },
    }
}
