#![cfg(metal_backend)]

use std::mem::size_of;

use bytemuck;
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use ndarray::{Array4, s};
use objc2::{rc::Retained, runtime::ProtocolObject};
use test_tag::tag;

use crate::{
    DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::{
                AttentionSinglePassKernel, AttentionTwoPass1Kernel, AttentionTwoPass2Kernel,
                attention::{AttentionGemmArguments, AttentionGemmBlock},
            },
        },
        metal::Metal,
    },
};

fn reference_attention(
    queries: &Array4<f32>, // [batch, num_heads, seq_len, head_dim]
    keys: &Array4<f32>,    // [batch, num_kv_heads, seq_len, head_dim]
    values: &Array4<f32>,  // [batch, num_kv_heads, seq_len, head_dim]
    sinks: Option<&[f32]>,
    scale: f32,
    is_causal: bool,
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

                    let q = scaled_queries.slice(s![b, q_head, .., ..]).to_owned(); // [L, D]
                    let k = keys.slice(s![b, kv_head, .., ..]).to_owned(); // [L_kv, D]
                    let v = values.slice(s![b, kv_head, .., ..]).to_owned(); // [L_kv, D]

                    // Efficient matrix multiplication: scores = Q @ K^T
                    let mut scores = q.dot(&k.t());

                    if is_causal {
                        for i in 0..seq_len {
                            for j in (i + 1)..kv_seq_len {
                                scores[[i, j]] = f32::NEG_INFINITY;
                            }
                        }
                    }

                    // Softmax with optional sink logit
                    for i in 0..seq_len {
                        let sink_logit = sinks.map(|s| s[q_head]);
                        let row = scores.row(i);
                        let max_score = row.iter().fold(sink_logit.unwrap_or(f32::NEG_INFINITY), |acc, &x| acc.max(x));

                        let mut sum_exp = sink_logit.map(|sink| (sink - max_score).exp()).unwrap_or(0.0);
                        for j in 0..kv_seq_len {
                            scores[[i, j]] = (scores[[i, j]] - max_score).exp();
                            sum_exp += scores[[i, j]];
                        }
                        scores.row_mut(i).mapv_inplace(|x| x / sum_exp);
                    }

                    // Efficient matrix multiplication: output = scores @ V
                    let head_output = scores.dot(&v);
                    output.slice_mut(s![b, q_head, .., ..]).assign(&head_output);
                }
            }
        } else {
            for h in 0..num_heads {
                let q = scaled_queries.slice(s![b, h, .., ..]).to_owned(); // [L, D]
                let k = keys.slice(s![b, h, .., ..]).to_owned(); // [L_kv, D]
                let v = values.slice(s![b, h, .., ..]).to_owned(); // [L_kv, D]

                // Efficient matrix multiplication: scores = Q @ K^T
                let mut scores = q.dot(&k.t());

                if is_causal {
                    for i in 0..seq_len {
                        for j in (i + 1)..kv_seq_len {
                            scores[[i, j]] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax with optional sink logit
                for i in 0..seq_len {
                    let sink_logit = sinks.map(|s| s[h]);
                    let row = scores.row(i);
                    let max_score = row.iter().fold(sink_logit.unwrap_or(f32::NEG_INFINITY), |acc, &x| acc.max(x));

                    let mut sum_exp = sink_logit.map(|sink| (sink - max_score).exp()).unwrap_or(0.0);
                    for j in 0..kv_seq_len {
                        scores[[i, j]] = (scores[[i, j]] - max_score).exp();
                        sum_exp += scores[[i, j]];
                    }
                    scores.row_mut(i).mapv_inplace(|x| x / sum_exp);
                }

                // Efficient matrix multiplication: output = scores @ V
                let head_output = scores.dot(&v);
                output.slice_mut(s![b, h, .., ..]).assign(&head_output);
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
) -> (Array4<f32>, Array4<f32>, Array4<f32>) {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(seed);

    let queries = Array4::from_shape_fn((batch_size, num_heads, seq_len, head_dim), |_| rng.random_range(-0.5..0.5));

    let keys = Array4::from_shape_fn((batch_size, num_kv_heads, seq_len, head_dim), |_| rng.random_range(-0.5..0.5));

    let values = Array4::from_shape_fn((batch_size, num_kv_heads, seq_len, head_dim), |_| rng.random_range(-0.5..0.5));

    (queries, keys, values)
}

/// Convert ndarray to Metal buffer layout expected by our kernel
fn create_query_buffer(
    queries: &Array4<f32>,
    context: &<Metal as Backend>::Context,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
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

    context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&query_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer")
}

fn create_key_cache_buffer(
    keys: &Array4<f32>,
    max_seq_len: usize,
    context: &<Metal as Backend>::Context,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let (_batch_size, num_kv_heads, seq_len, head_dim) = keys.dim();

    // Our kernel expects key cache layout: [max_seq_len, num_kv_heads, head_dim]
    let mut key_cache_data = vec![0.0f32; max_seq_len * num_kv_heads * head_dim];

    for t in 0..seq_len {
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                let idx = t * num_kv_heads * head_dim + h * head_dim + d;
                key_cache_data[idx] = keys[[0, h, t, d]];
            }
        }
    }

    context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&key_cache_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer")
}

fn create_value_cache_buffer(
    values: &Array4<f32>,
    max_seq_len: usize,
    context: &<Metal as Backend>::Context,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let (_batch_size, num_kv_heads, seq_len, head_dim) = values.dim();

    // Our kernel expects value cache layout: [max_seq_len, num_kv_heads, head_dim]
    let mut value_cache_data = vec![0.0f32; max_seq_len * num_kv_heads * head_dim];

    for t in 0..seq_len {
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                let idx = t * num_kv_heads * head_dim + h * head_dim + d;
                value_cache_data[idx] = values[[0, h, t, d]];
            }
        }
    }

    context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&value_cache_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer")
}

fn create_sinks_buffer(
    sinks: &[f32],
    context: &<Metal as Backend>::Context,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&sinks), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer")
}

fn convert_kernel_output(
    output_slice: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Array4<f32> {
    let mut kernel_output = Array4::zeros((batch_size, num_heads, seq_len, head_dim));
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
    kernel: &<<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel,
    context: &<Metal as Backend>::Context,
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    sinks: Option<&[f32]>,
    scale: f32,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_batch_size, num_kv_heads, _seq_len, _head_dim) = keys.dim();

    let query_buffer = create_query_buffer(queries, context);
    let key_cache_buffer = create_key_cache_buffer(keys, seq_len, context);
    let value_cache_buffer = create_value_cache_buffer(values, seq_len, context);

    let sinks_buffer = sinks.map(|s| create_sinks_buffer(s, context));

    let mut output_buffer = context
        .device
        .new_buffer(num_heads * seq_len * head_dim * size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    kernel.encode(
        &query_buffer,
        &key_cache_buffer,
        &value_cache_buffer,
        &mut output_buffer,
        (num_heads / num_kv_heads) as u32,
        seq_len as u32,
        head_dim as u32,
        (num_kv_heads * head_dim) as u32,
        head_dim as u32,
        (num_kv_heads * head_dim) as u32,
        None,
        scale,
        None::<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        None,
        sinks_buffer.as_ref().map(|b| b),
        num_heads as u32,
        seq_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let output_ptr = output_buffer.contents().as_ptr() as *const f32;
    let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, num_heads * seq_len * head_dim) };

    let kernel_output = convert_kernel_output(output_slice, batch_size, num_heads, seq_len, head_dim);

    Ok(kernel_output)
}

fn run_single_pass_attention_with_is_causal(
    kernel: &<<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel,
    context: &<Metal as Backend>::Context,
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    sinks: Option<&[f32]>,
    scale: f32,
    _is_causal: bool,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_batch_size, num_kv_heads, _seq_len, _head_dim) = keys.dim();

    let query_buffer = create_query_buffer(queries, context);
    let key_cache_buffer = create_key_cache_buffer(keys, seq_len, context);
    let value_cache_buffer = create_value_cache_buffer(values, seq_len, context);

    let sinks_buffer = sinks.map(|s| create_sinks_buffer(s, context));

    let mut output_buffer = context
        .device
        .new_buffer(num_heads * seq_len * head_dim * size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    kernel.encode(
        &query_buffer,
        &key_cache_buffer,
        &value_cache_buffer,
        &mut output_buffer,
        (num_heads / num_kv_heads) as u32,
        seq_len as u32,
        head_dim as u32,
        (num_kv_heads * head_dim) as u32,
        head_dim as u32,
        (num_kv_heads * head_dim) as u32,
        None,
        scale,
        None::<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        None,
        sinks_buffer.as_ref().map(|b| b),
        num_heads as u32,
        seq_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let output_ptr = output_buffer.contents().as_ptr() as *const f32;
    let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, num_heads * seq_len * head_dim) };

    let kernel_output = convert_kernel_output(output_slice, batch_size, num_heads, seq_len, head_dim);

    Ok(kernel_output)
}

fn run_gemm_attention(
    kernel: &AttentionGemmBlock<Metal>,
    context: &<Metal as Backend>::Context,
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    sinks: Option<&[f32]>,
    scale: f32,
    is_causal: bool,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_batch_size, num_kv_heads, _seq_len, _head_dim) = keys.dim();

    let query_buffer = create_query_buffer(queries, context);
    let key_cache_buffer = create_key_cache_buffer(keys, seq_len, context);
    let value_cache_buffer = create_value_cache_buffer(values, seq_len, context);

    let sinks_buffer = sinks.map(|s| create_sinks_buffer(s, context));

    let mut output_buffer = context
        .device
        .new_buffer(num_heads * seq_len * head_dim * size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    let args = AttentionGemmArguments {
        queries_buffer: &query_buffer,
        keys_buffer: &key_cache_buffer,
        values_buffer: &value_cache_buffer,
        output_buffer: &mut output_buffer,
        trie_buffer: None,
        sinks_buffer: sinks_buffer.as_ref(),
        num_heads,
        num_groups: num_kv_heads,
        suffix_length: seq_len,
        sequence_length: seq_len,
        segment_prefix_length: 0,
        max_sequence_length: seq_len,
        ring_params: None,
        head_dim,
        sliding_window_size: None,
        is_causal,
        scale,
        k_head_stride: head_dim as u64,
        k_seq_stride: (num_kv_heads * head_dim) as u64,
        v_head_stride: head_dim as u64,
        v_seq_stride: (num_kv_heads * head_dim) as u64,
    };

    kernel.encode(context, &mut encoder, args)?;
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let output_ptr = output_buffer.contents().as_ptr() as *const f32;
    let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, num_heads * seq_len * head_dim) };

    let kernel_output = convert_kernel_output(output_slice, batch_size, num_heads, seq_len, head_dim);

    Ok(kernel_output)
}

fn compare_results(
    kernel_output: &Array4<f32>,
    reference_output: &Array4<f32>,
    tolerance: f32,
    test_name: &str,
) -> Result<(), String> {
    let max_diff = kernel_output.iter().zip(reference_output.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

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
    println!("Kernel output [0,0,0,0:4]: {:?}", &kernel_output.slice(s![0, 0, 0, 0..4]).to_vec());
    println!("Reference output [0,0,0,0:4]: {:?}", &reference_output.slice(s![0, 0, 0, 0..4]).to_vec());
    println!("Kernel output [0,0,1,0:4]: {:?}", &kernel_output.slice(s![0, 0, 1, 0..4]).to_vec());
    println!("Reference output [0,0,1,0:4]: {:?}", &reference_output.slice(s![0, 0, 1, 0..4]).to_vec());

    let kernel_max = kernel_output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let kernel_min = kernel_output.iter().fold(f32::INFINITY, |a, &b| a.min(b.abs()));
    let ref_max = reference_output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let ref_min = reference_output.iter().fold(f32::INFINITY, |a, &b| a.min(b.abs()));

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
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 4;
    let num_kv_heads = 4;
    let seq_len = 8;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 42);

    println!("Testing reference implementation...");
    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);

    let ref_max = reference_output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let ref_min = reference_output.iter().fold(f32::INFINITY, |a, &b| a.min(b.abs()));
    println!("Reference output range: [{}, {}]", ref_min, ref_max);
    println!("Reference sample values: {:?}", &reference_output.slice(s![0, 0, 0, 0..4]).to_vec());

    let is_causal = false; // Non-causal attention for this test
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
        false,
        false,
        is_causal,
        false,
        false,
    )
    .expect("Failed to create attention single pass metal");

    let kernel_output = match run_single_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run single-pass attention: {:?}", e);
        },
    };

    let tolerance = 1e-2;
    if let Err(e) = compare_results(&kernel_output, &reference_output, tolerance, "Single-pass attention") {
        panic!("{}", e);
    }
}

#[test]
fn test_gemm_attention_basic() {
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 4;
    let num_kv_heads = 4;
    let seq_len = 32;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 123);

    let kernel = AttentionGemmBlock::new(DataType::F32);

    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);

    let kernel_output =
        run_gemm_attention(&kernel, &context, &queries, &keys, &values, None, scale, /*is_causal=*/ false)
            .expect("run gemm attention");

    let tolerance = 1e-2;
    if let Err(e) = compare_results(&kernel_output, &reference_output, tolerance, "Gemm attention") {
        panic!("{}", e);
    }
}

#[test]
fn test_gemm_attention_f32_head_dim_128() {
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 4;
    let num_kv_heads = 4;
    let seq_len = 32;
    let head_dim = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 456);

    let kernel = AttentionGemmBlock::new(DataType::F32);

    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);

    let kernel_output =
        run_gemm_attention(&kernel, &context, &queries, &keys, &values, None, scale, /*is_causal=*/ false)
            .expect("run gemm attention f32 head_dim=128");

    let tolerance = 1e-2;
    if let Err(e) = compare_results(&kernel_output, &reference_output, tolerance, "Gemm attention f32 head_dim=128") {
        panic!("{}", e);
    }
}

#[test]
fn test_matrix_attention_matches_vector_and_cpu_seq256() {
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 8;
    let num_kv_heads = 2;
    let seq_len = 256;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let is_causal = true;

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 2026);

    // CPU reference
    let reference_output = reference_attention(&queries, &keys, &values, None, scale, is_causal);

    // vector attention path (single-pass)
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
        false,
        false,
        is_causal,
        false,
        false,
    )
    .expect("Failed to create attention single pass metal");
    let vector_output =
        run_single_pass_attention_with_is_causal(&kernel, &context, &queries, &keys, &values, None, scale, is_causal)
            .expect("run vector attention");

    // matrix attention path (gemm)
    let gemm_kernel = AttentionGemmBlock::new(DataType::F32);
    let matrix_output = run_gemm_attention(&gemm_kernel, &context, &queries, &keys, &values, None, scale, is_causal)
        .expect("run matrix attention");

    // Compare both to CPU
    let tol_cpu = 5e-2;
    compare_results(&vector_output, &reference_output, tol_cpu, "vector single-pass attention vs CPU").unwrap();
    compare_results(&matrix_output, &reference_output, tol_cpu, "matrix attention vs CPU").unwrap();

    let max_diff_vector_matrix =
        vector_output.iter().zip(matrix_output.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let tol_vector_matrix = 5e-2;
    println!("Max absolute difference (vector single-pass attention vs matrix attention): {}", max_diff_vector_matrix);
    assert!(
        max_diff_vector_matrix <= tol_vector_matrix,
        "vector single-pass attention and matrix attention differ too much: max_diff = {} (tol={})",
        max_diff_vector_matrix,
        tol_vector_matrix
    );
}

#[test]
fn test_single_pass_attention_with_sinks() {
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 4;
    let num_kv_heads = 4;
    let seq_len = 8;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 444);

    let sinks: Vec<f32> = (0..num_heads).map(|h| (h as f32 - (num_heads as f32 / 2.0)) * 0.25).collect();
    println!("Using sinks: {:?}", sinks);

    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
        true,
        false,
        false,
        false,
        false,
    )
    .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel");

    let reference_output = reference_attention(&queries, &keys, &values, Some(&sinks), scale, false);

    let kernel_output =
        match run_single_pass_attention(&kernel, &context, &queries, &keys, &values, Some(&sinks), scale) {
            Ok(output) => output,
            Err(e) => {
                panic!("Failed to run single-pass attention with sinks: {:?}", e);
            },
        };

    let tolerance = 1e-2;
    if let Err(e) = compare_results(&kernel_output, &reference_output, tolerance, "Single-pass attention with sinks") {
        panic!("{}", e);
    }
}

#[test]
fn test_single_pass_attention_with_sinks_long_sequence() {
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 4;
    let num_kv_heads = 4;
    let seq_len = 64;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 777);

    let sinks: Vec<f32> = (0..num_heads).map(|h| (h as f32 * 0.1) - 0.15).collect();
    println!("Long sequence sinks: {:?}", sinks);

    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
        true,
        false,
        false,
        false,
        false,
    )
    .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel");

    let reference_output = reference_attention(&queries, &keys, &values, Some(&sinks), scale, false);

    let kernel_output =
        match run_single_pass_attention(&kernel, &context, &queries, &keys, &values, Some(&sinks), scale) {
            Ok(output) => output,
            Err(e) => {
                panic!("Failed to run single-pass attention long sequence with sinks: {:?}", e);
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
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 8;
    let num_kv_heads = 2;
    let seq_len = 8;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 42);

    let is_causal = false; // Non-causal attention for this test
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
        false,
        false,
        is_causal,
        false,
        false,
    )
    .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel");

    let kernel_output = match run_single_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale) {
        Ok(output) => output,
        Err(e) => {
            panic!("Failed to run single-pass attention GQA: {:?}", e);
        },
    };

    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);

    let tolerance = 1e-2;
    if let Err(e) = compare_results(&kernel_output, &reference_output, tolerance, "Single-pass attention GQA") {
        panic!("{}", e);
    }
}

fn run_two_pass_attention(
    kernel_pass1: &<<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel,
    kernel_pass2: &<<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel,
    context: &<Metal as Backend>::Context,
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    sinks: Option<&[f32]>,
    scale: f32,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_, num_kv_heads, _, _) = keys.dim();

    let queries_buffer = create_query_buffer(queries, context);
    let keys_buffer = create_key_cache_buffer(keys, seq_len, context);
    let values_buffer = create_value_cache_buffer(values, seq_len, context);
    let sinks_buffer = sinks.map(|s| create_sinks_buffer(s, context));

    let total_blocks_count = 32;
    let partials_size = num_heads * seq_len * total_blocks_count * head_dim;
    let sums_maxs_size = num_heads * seq_len * total_blocks_count;

    let mut partials_buffer = context
        .device
        .new_buffer(partials_size * std::mem::size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut sums_buffer = context
        .device
        .new_buffer(sums_maxs_size * std::mem::size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut maxs_buffer = context
        .device
        .new_buffer(sums_maxs_size * std::mem::size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut output_buffer = context
        .device
        .new_buffer(
            num_heads * seq_len * head_dim * std::mem::size_of::<f32>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    kernel_pass1.encode(
        &queries_buffer,
        &keys_buffer,
        &values_buffer,
        &mut partials_buffer,
        &mut sums_buffer,
        &mut maxs_buffer,
        (num_heads / num_kv_heads) as u32,
        seq_len as u32,
        head_dim as u32,
        (num_kv_heads * head_dim) as u32,
        head_dim as u32,
        (num_kv_heads * head_dim) as u32,
        None,
        scale,
        num_heads as u32,
        seq_len as u32,
        None::<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        None,
        sinks_buffer.as_ref().map(|b| b),
        &mut encoder,
    );
    kernel_pass2.encode(
        &partials_buffer,
        &sums_buffer,
        &maxs_buffer,
        &mut output_buffer,
        num_heads as u32,
        seq_len as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let output_ptr = output_buffer.contents().as_ptr() as *const f32;
    let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, num_heads * seq_len * head_dim) };

    let kernel_output = convert_kernel_output(output_slice, batch_size, num_heads, seq_len, head_dim);

    Ok(kernel_output)
}

#[test]
fn test_two_pass_attention() {
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 8;
    let num_kv_heads: usize = 8;
    let seq_len = 2048;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let is_causal = false; // Non-causal attention for this test

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 42);

    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);

    let kernel_pass1 = <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
        false,
        false,
        is_causal,
        false,
        false,
    )
    .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel");
    let kernel_pass2 = <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
    )
    .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel");
    let kernel_output =
        match run_two_pass_attention(&kernel_pass1, &kernel_pass2, &context, &queries, &keys, &values, None, scale) {
            Ok(output) => output,
            Err(e) => {
                panic!("Failed to run two-pass attention: {:?}", e);
            },
        };

    let tolerance = 1e-2;
    if let Err(e) = compare_results(&kernel_output, &reference_output, tolerance, "Two-pass attention") {
        panic!("{}", e);
    }
}

#[test]
fn test_two_pass_attention_gqa() {
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 8;
    let num_kv_heads = 2;
    let seq_len = 4096;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let is_causal = false; // Non-causal attention for this test

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 42);

    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);

    let kernel_pass1 = <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
        false,
        false,
        is_causal,
        false,
        false,
    )
    .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel");
    let kernel_pass2 = <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
    )
    .expect("Failed to create <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel");
    let kernel_output =
        match run_two_pass_attention(&kernel_pass1, &kernel_pass2, &context, &queries, &keys, &values, None, scale) {
            Ok(output) => output,
            Err(e) => {
                panic!("Failed to run two-pass attention GQA: {:?}", e);
            },
        };

    let tolerance = 1e-2;
    if let Err(e) = compare_results(&kernel_output, &reference_output, tolerance, "Two-pass attention GQA") {
        panic!("{}", e);
    }
}

#[tag(heavy)]
#[test]
fn perf_two_pass_attention() {
    use std::time::Instant;

    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    // ---- Problem sizes requiring two-pass ----
    let batch_size = 1;
    let num_heads = 32;
    let num_kv_heads = 32;
    let seq_len = 8192; // Large sequence length (prefix + suffix)
    let suffix_length = 1; // Only processing 1 new token (realistic inference)
    let head_dim = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let is_causal = false;

    println!(
        "Creating test data for two-pass performance test (prefix={}, suffix={})...",
        seq_len - suffix_length,
        suffix_length
    );
    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 123);

    let kernel_pass1 = <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
        false,
        false,
        is_causal,
        false,
        false,
    )
    .expect("Failed to create AttentionTwoPass1Kernel");
    let kernel_pass2 = <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel::new(
        &context,
        DataType::F32,
        head_dim as u32,
    )
    .expect("Failed to create AttentionTwoPass2Kernel");

    // ---- Create buffers ----
    // For realistic inference, we only process queries for the suffix (new tokens)
    let queries_suffix = queries.slice(s![.., .., (seq_len - suffix_length).., ..]).to_owned();
    let queries_buffer = create_query_buffer(&queries_suffix, &context);
    let keys_buffer = create_key_cache_buffer(&keys, seq_len, &context);
    let values_buffer = create_value_cache_buffer(&values, seq_len, &context);

    let total_blocks_count = 32;
    let partials_size = num_heads * suffix_length * total_blocks_count * head_dim;
    let sums_maxs_size = num_heads * suffix_length * total_blocks_count;

    let mut partials_buffer = context
        .device
        .new_buffer(partials_size * std::mem::size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let mut sums_buffer = context
        .device
        .new_buffer(sums_maxs_size * std::mem::size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let mut maxs_buffer = context
        .device
        .new_buffer(sums_maxs_size * std::mem::size_of::<f32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let mut output_buffer = context
        .device
        .new_buffer(
            num_heads * suffix_length * head_dim * std::mem::size_of::<f32>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");

    // ---- Launch and time ----
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    let sinks_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>> = None;
    kernel_pass1.encode(
        &queries_buffer,
        &keys_buffer,
        &values_buffer,
        &mut partials_buffer,
        &mut sums_buffer,
        &mut maxs_buffer,
        (num_heads / num_kv_heads) as u32,
        seq_len as u32,
        head_dim as u32,
        (num_kv_heads * head_dim) as u32,
        head_dim as u32,
        (num_kv_heads * head_dim) as u32,
        None,
        scale,
        num_heads as u32,
        suffix_length as u32,
        None::<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        None,
        sinks_buffer.as_ref().map(|b| b),
        &mut encoder,
    );
    kernel_pass2.encode(
        &partials_buffer,
        &sums_buffer,
        &maxs_buffer,
        &mut output_buffer,
        num_heads as u32,
        suffix_length as u32,
        &mut encoder,
    );
    // Time both host-side and GPU execution
    let host_timer = Instant::now();
    let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    let gpu_time_ms = completed.gpu_execution_time().as_secs_f64() * 1e3;
    println!(
        "Two-pass attention perf (heads={}, prefix={}, suffix={}, head_dim={}): GPU={:.2} ms, Host-side={:.2} ms",
        num_heads,
        seq_len - suffix_length,
        suffix_length,
        head_dim,
        gpu_time_ms,
        host_elapsed_ms
    );

    // ---- Sanity check ----
    let output_ptr = output_buffer.contents().as_ptr() as *const f32;
    let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, num_heads * suffix_length * head_dim) };

    // Check for NaN/Inf
    for &val in output_slice.iter().take(100) {
        assert!(val.is_finite(), "Output contains non-finite values");
    }

    println!("✓ Two-pass attention performance test completed");
}
