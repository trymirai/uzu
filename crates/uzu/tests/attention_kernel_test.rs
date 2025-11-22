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
    sinks: Option<&[f32]>,
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
                        let sink_logit = sinks.map(|s| s[q_head]);
                        let mut max_score =
                            sink_logit.unwrap_or(f32::NEG_INFINITY);
                        for j in 0..kv_seq_len {
                            max_score = max_score.max(scores[[i, j]]);
                        }

                        let mut sum_exp = sink_logit
                            .map(|sink| (sink - max_score).exp())
                            .unwrap_or(0.0);
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
                    let sink_logit = sinks.map(|s| s[h]);
                    let mut max_score = sink_logit.unwrap_or(f32::NEG_INFINITY);
                    for j in 0..kv_seq_len {
                        max_score = max_score.max(scores[[i, j]]);
                    }

                    let mut sum_exp = sink_logit
                        .map(|sink| (sink - max_score).exp())
                        .unwrap_or(0.0);
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

fn create_sinks_buffer(
    sinks: &[f32],
    context: &MTLContext,
) -> metal::Buffer {
    context.device.new_buffer_with_data(
        sinks.as_ptr() as *const _,
        (sinks.len() * size_of::<f32>()) as u64,
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
    sinks: Option<&[f32]>,
    scale: f32,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_batch_size, num_kv_heads, _seq_len, _head_dim) = keys.dim();

    let query_buffer = create_query_buffer(queries, context);
    let key_cache_buffer = create_key_cache_buffer(keys, seq_len, context);
    let value_cache_buffer =
        create_value_cache_buffer(values, seq_len, context);

    let mask_buffer = mask.map(|m| create_mask_buffer(m, num_heads, context));
    let sinks_buffer = sinks.map(|s| create_sinks_buffer(s, context));

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
        sinks_buffer: sinks_buffer.as_ref(),
        num_heads,
        suffix_length: seq_len,
        head_dim,
        is_causal: false,
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
        reference_attention(&queries, &keys, &values, None, None, scale);

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

    let is_causal = false; // Non-causal attention for this test
    let variant = kernel.choose_variant(seq_len, head_dim, is_causal);
    println!("Using kernel variant: {:?}", variant);
    println!(
        "Supports single-pass for head_dim={}: {}",
        head_dim,
        kernel.supports_single_pass(head_dim, is_causal)
    );
    println!(
        "Supports two-pass for head_dim={}: {}",
        head_dim,
        kernel.supports_two_pass(head_dim, is_causal)
    );

    if !kernel.supports_single_pass(head_dim, false) {
        panic!("Single-pass kernel not supported for head_dim={}", head_dim);
    }

    let kernel_output = match run_single_pass_attention(
        &kernel, &context, &queries, &keys, &values, None, None, scale,
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

    if !kernel.supports_single_pass(head_dim, false) {
        panic!("Single-pass kernel not supported for head_dim={}", head_dim);
    }

    let kernel_output = match run_single_pass_attention(
        &kernel,
        &context,
        &queries,
        &keys,
        &values,
        Some(&mask),
        None,
        scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run single-pass attention with mask: {:?}", e);
        },
    };

    let reference_output =
        reference_attention(&queries, &keys, &values, Some(&mask), None, scale);

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
fn test_single_pass_attention_with_sinks() {
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
        444,
    );

    let sinks: Vec<f32> = (0..num_heads)
        .map(|h| (h as f32 - (num_heads as f32 / 2.0)) * 0.25)
        .collect();
    println!("Using sinks: {:?}", sinks);

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    if !kernel.supports_single_pass(head_dim, false) {
        panic!("Single-pass kernel not supported for head_dim={}", head_dim);
    }

    let reference_output = reference_attention(
        &queries,
        &keys,
        &values,
        None,
        Some(&sinks),
        scale,
    );

    let kernel_output = match run_single_pass_attention(
        &kernel,
        &context,
        &queries,
        &keys,
        &values,
        None,
        Some(&sinks),
        scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run single-pass attention with sinks: {:?}", e);
        },
    };

    let tolerance = 1e-2;
    if let Err(e) = compare_results(
        &kernel_output,
        &reference_output,
        tolerance,
        "Single-pass attention with sinks",
    ) {
        panic!("{}", e);
    }
}

#[test]
fn test_single_pass_attention_with_sinks_long_sequence() {
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
    let seq_len = 64;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values, _mask) = create_test_data(
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        777,
    );

    let sinks: Vec<f32> =
        (0..num_heads).map(|h| (h as f32 * 0.1) - 0.15).collect();
    println!("Long sequence sinks: {:?}", sinks);

    let kernel = match AttentionKernel::new(&context, KernelDataType::Float32) {
        Ok(k) => k,
        Err(e) => {
            panic!("Failed to create AttentionKernel: {:?}", e);
        },
    };

    if !kernel.supports_single_pass(head_dim, false) {
        panic!("Single-pass kernel not supported for head_dim={}", head_dim);
    }

    let reference_output = reference_attention(
        &queries,
        &keys,
        &values,
        None,
        Some(&sinks),
        scale,
    );

    let kernel_output = match run_single_pass_attention(
        &kernel,
        &context,
        &queries,
        &keys,
        &values,
        None,
        Some(&sinks),
        scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!(
                "Failed to run single-pass attention long sequence with sinks: {:?}",
                e
            );
        },
    };

    let tolerance = 5e-2;
    if let Err(e) = compare_results(
        &kernel_output,
        &reference_output,
        tolerance,
        "Single-pass attention with sinks (long sequence)",
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

    let is_causal = false; // Non-causal attention for this test
    let variant = kernel.choose_variant(seq_len, head_dim, is_causal);
    println!("Using kernel variant for GQA: {:?}", variant);

    if !kernel.supports_single_pass(head_dim, false) {
        panic!("Single-pass kernel not supported for head_dim={}", head_dim);
    }

    let kernel_output = match run_single_pass_attention(
        &kernel, &context, &queries, &keys, &values, None, None, scale,
    ) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run single-pass attention GQA: {:?}", e);
        },
    };

    let reference_output =
        reference_attention(&queries, &keys, &values, None, None, scale);

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
    sinks: Option<&[f32]>,
    scale: f32,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_, num_kv_heads, _, _) = keys.dim();

    let queries_buffer = create_query_buffer(queries, context);
    let keys_buffer = create_key_cache_buffer(keys, seq_len, context);
    let values_buffer = create_value_cache_buffer(values, seq_len, context);
    let mask_buffer = mask.map(|m| create_mask_buffer(m, num_heads, context));
    let sinks_buffer = sinks.map(|s| create_sinks_buffer(s, context));

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
        sinks_buffer: sinks_buffer.as_ref(),
        num_heads,
        suffix_length: seq_len,
        head_dim,
        is_causal: false,
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
    let is_causal = false; // Non-causal attention for this test

    if !kernel.supports_two_pass(head_dim, is_causal) {
        panic!("Two-pass kernel not supported for head_dim={}", head_dim);
    }

    let variant = kernel.choose_variant(seq_len, head_dim, is_causal);
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
        reference_attention(&queries, &keys, &values, None, None, scale);

    let kernel_output = match run_two_pass_attention(
        &kernel, &context, &queries, &keys, &values, None, None, scale,
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
    let is_causal = false; // Non-causal attention for this test

    if !kernel.supports_two_pass(head_dim, is_causal) {
        panic!("Two-pass kernel not supported for head_dim={}", head_dim);
    }

    let variant = kernel.choose_variant(seq_len, head_dim, is_causal);
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
        reference_attention(&queries, &keys, &values, None, None, scale);

    let kernel_output = match run_two_pass_attention(
        &kernel, &context, &queries, &keys, &values, None, None, scale,
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
#[ignore]
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
    let is_causal = false;

    if !kernel.supports_two_pass(head_dim, is_causal) {
        println!(
            "Skipping two-pass perf test: not supported for head_dim={}",
            head_dim
        );
        return;
    }

    let variant = kernel.choose_variant(seq_len, head_dim, is_causal);
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
        sinks_buffer: None,
        num_heads,
        suffix_length, // Use actual suffix_length, not seq_len
        head_dim,
        is_causal: false,
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
