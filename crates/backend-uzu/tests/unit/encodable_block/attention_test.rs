#![cfg(metal_backend)]

use std::mem::size_of;

use ndarray::{Array4, s};
use test_tag::tag;

use crate::{
    DataType, allocation_to_vec,
    backends::{
        common::{
            Allocation, AllocationType, Backend, Context, Encoder, Kernels,
            kernel::{
                AttentionSinglePassKernel, AttentionTwoPass1Kernel, AttentionTwoPass2Kernel,
                attention::{AttentionGemmArguments, AttentionGemmBlock},
            },
        },
        metal::Metal,
    },
};

fn alloc_allocation_with_data<T: crate::ArrayElement>(
    context: &<Metal as Backend>::Context,
    data: &[T],
) -> Allocation<Metal> {
    let allocation = context
        .create_allocation(data.len() * size_of::<T>(), AllocationType::Global)
        .expect("Failed to create allocation");
    crate::allocation_copy_from_slice(&allocation, data).expect("Failed to initialize allocation");
    allocation
}

fn alloc_allocation<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> Allocation<B> {
    context
        .create_allocation(elements_count * size_of::<T>(), crate::backends::common::AllocationType::Global)
        .expect("Failed to create allocation")
}

fn submit_encoder<B: Backend>(encoder: Encoder<B>) {
    encoder.end_encoding().submit().wait_until_completed().unwrap();
}

fn reference_attention(
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    sinks: Option<&[f32]>,
    scale: f32,
    is_causal: bool,
) -> Array4<f32> {
    let (batch_size, num_heads, sequence_length, head_dim) = queries.dim();
    let (_, num_kv_heads, kv_sequence_length, _) = keys.dim();
    let repeats_per_kv_head = num_heads / num_kv_heads;
    let scaled_queries = queries.mapv(|value| value * scale);
    let mut output = Array4::zeros((batch_size, num_heads, sequence_length, head_dim));

    for batch_index in 0..batch_size {
        if repeats_per_kv_head > 1 {
            for kv_head_index in 0..num_kv_heads {
                for repeat_index in 0..repeats_per_kv_head {
                    let head_index = kv_head_index * repeats_per_kv_head + repeat_index;
                    assign_attention_head_output(
                        &mut output,
                        &scaled_queries,
                        keys,
                        values,
                        sinks,
                        is_causal,
                        batch_index,
                        head_index,
                        kv_head_index,
                        sequence_length,
                        kv_sequence_length,
                    );
                }
            }
        } else {
            for head_index in 0..num_heads {
                assign_attention_head_output(
                    &mut output,
                    &scaled_queries,
                    keys,
                    values,
                    sinks,
                    is_causal,
                    batch_index,
                    head_index,
                    head_index,
                    sequence_length,
                    kv_sequence_length,
                );
            }
        }
    }

    output
}

fn assign_attention_head_output(
    output: &mut Array4<f32>,
    scaled_queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    sinks: Option<&[f32]>,
    is_causal: bool,
    batch_index: usize,
    head_index: usize,
    kv_head_index: usize,
    sequence_length: usize,
    kv_sequence_length: usize,
) {
    let queries = scaled_queries.slice(s![batch_index, head_index, .., ..]).to_owned();
    let keys = keys.slice(s![batch_index, kv_head_index, .., ..]).to_owned();
    let values = values.slice(s![batch_index, kv_head_index, .., ..]).to_owned();
    let mut scores = queries.dot(&keys.t());

    if is_causal {
        for query_index in 0..sequence_length {
            for key_index in (query_index + 1)..kv_sequence_length {
                scores[[query_index, key_index]] = f32::NEG_INFINITY;
            }
        }
    }

    for query_index in 0..sequence_length {
        let sink_logit = sinks.map(|sink_values| sink_values[head_index]);
        let row = scores.row(query_index);
        let max_score =
            row.iter().fold(sink_logit.unwrap_or(f32::NEG_INFINITY), |current_max, &score| current_max.max(score));

        let mut sum_exp = sink_logit.map(|sink_value| (sink_value - max_score).exp()).unwrap_or(0.0);
        for key_index in 0..kv_sequence_length {
            scores[[query_index, key_index]] = (scores[[query_index, key_index]] - max_score).exp();
            sum_exp += scores[[query_index, key_index]];
        }
        scores.row_mut(query_index).mapv_inplace(|value| value / sum_exp);
    }

    let head_output = scores.dot(&values);
    output.slice_mut(s![batch_index, head_index, .., ..]).assign(&head_output);
}

fn create_test_data(
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    sequence_length: usize,
    head_dim: usize,
    seed: u64,
) -> (Array4<f32>, Array4<f32>, Array4<f32>) {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    let mut random = StdRng::seed_from_u64(seed);

    let queries =
        Array4::from_shape_fn((batch_size, num_heads, sequence_length, head_dim), |_| random.random_range(-0.5..0.5));
    let keys = Array4::from_shape_fn((batch_size, num_kv_heads, sequence_length, head_dim), |_| {
        random.random_range(-0.5..0.5)
    });
    let values = Array4::from_shape_fn((batch_size, num_kv_heads, sequence_length, head_dim), |_| {
        random.random_range(-0.5..0.5)
    });

    (queries, keys, values)
}

fn create_query_allocation(
    queries: &Array4<f32>,
    context: &<Metal as Backend>::Context,
) -> Allocation<Metal> {
    let (_batch_size, num_heads, sequence_length, head_dim) = queries.dim();
    let mut values = vec![0.0_f32; num_heads * sequence_length * head_dim];

    for head_index in 0..num_heads {
        for sequence_index in 0..sequence_length {
            for dim_index in 0..head_dim {
                let flat_index = head_index * sequence_length * head_dim + sequence_index * head_dim + dim_index;
                values[flat_index] = queries[[0, head_index, sequence_index, dim_index]];
            }
        }
    }

    alloc_allocation_with_data(context, &values)
}

fn create_attention_cache_allocation(
    values: &Array4<f32>,
    max_sequence_length: usize,
    context: &<Metal as Backend>::Context,
) -> Allocation<Metal> {
    let (_batch_size, num_kv_heads, sequence_length, head_dim) = values.dim();
    let mut cache = vec![0.0_f32; max_sequence_length * num_kv_heads * head_dim];

    for sequence_index in 0..sequence_length {
        for head_index in 0..num_kv_heads {
            for dim_index in 0..head_dim {
                let flat_index = sequence_index * num_kv_heads * head_dim + head_index * head_dim + dim_index;
                cache[flat_index] = values[[0, head_index, sequence_index, dim_index]];
            }
        }
    }

    alloc_allocation_with_data(context, &cache)
}

fn create_sinks_allocation(
    sinks: &[f32],
    context: &<Metal as Backend>::Context,
) -> Allocation<Metal> {
    alloc_allocation_with_data(context, sinks)
}

fn convert_kernel_output(
    output: &[f32],
    batch_size: usize,
    num_heads: usize,
    sequence_length: usize,
    head_dim: usize,
) -> Array4<f32> {
    let mut kernel_output = Array4::zeros((batch_size, num_heads, sequence_length, head_dim));

    for head_index in 0..num_heads {
        for sequence_index in 0..sequence_length {
            for dim_index in 0..head_dim {
                let flat_index = (sequence_index * num_heads + head_index) * head_dim + dim_index;
                kernel_output[[0, head_index, sequence_index, dim_index]] = output[flat_index];
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

    let query_buffer = create_query_allocation(queries, context);
    let key_cache_buffer = create_attention_cache_allocation(keys, seq_len, context);
    let value_cache_buffer = create_attention_cache_allocation(values, seq_len, context);
    let sinks_buffer = sinks.map(|sinks| create_sinks_allocation(sinks, context));
    let mut output_buffer = alloc_allocation::<Metal, f32>(context, num_heads * seq_len * head_dim);

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
        None::<&Allocation<Metal>>,
        None,
        sinks_buffer.as_ref().map(|b| b),
        num_heads as u32,
        seq_len as u32,
        &mut encoder,
    );
    submit_encoder(encoder);

    let output_slice: Vec<f32> = allocation_to_vec(&output_buffer);
    let kernel_output = convert_kernel_output(&output_slice, batch_size, num_heads, seq_len, head_dim);

    Ok(kernel_output)
}

fn create_single_pass_kernel(
    context: &<Metal as Backend>::Context,
    head_dim: usize,
    has_sinks: bool,
    is_causal: bool,
) -> <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel {
    <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        context,
        DataType::F32,
        head_dim as u32,
        has_sinks,
        false,
        is_causal,
        false,
        false,
    )
    .expect("Failed to create attention single pass kernel")
}

fn create_two_pass_kernels(
    context: &<Metal as Backend>::Context,
    head_dim: usize,
    is_causal: bool,
) -> (
    <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel,
    <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel,
) {
    (
        <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
            context,
            DataType::F32,
            head_dim as u32,
            false,
            false,
            is_causal,
            false,
            false,
        )
        .expect("Failed to create AttentionTwoPass1Kernel"),
        <<Metal as Backend>::Kernels as Kernels>::AttentionTwoPass2Kernel::new(context, DataType::F32, head_dim as u32)
            .expect("Failed to create AttentionTwoPass2Kernel"),
    )
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

    let query_allocation = create_query_allocation(queries, context);
    let key_allocation = create_attention_cache_allocation(keys, seq_len, context);
    let value_allocation = create_attention_cache_allocation(values, seq_len, context);

    let sinks_allocation = sinks.map(|sinks| create_sinks_allocation(sinks, context));

    let mut output_allocation = alloc_allocation::<Metal, f32>(context, num_heads * seq_len * head_dim);

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    let args = AttentionGemmArguments {
        queries_buffer: &query_allocation,
        keys_buffer: &key_allocation,
        values_buffer: &value_allocation,
        output_buffer: &mut output_allocation,
        trie_buffer: None,
        sinks_buffer: sinks_allocation.as_ref(),
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

    kernel.encode(&mut encoder, args)?;
    submit_encoder(encoder);

    let output: Vec<f32> = allocation_to_vec(&output_allocation);

    let kernel_output = convert_kernel_output(&output, batch_size, num_heads, seq_len, head_dim);

    Ok(kernel_output)
}

fn compare_results(
    kernel_output: &Array4<f32>,
    reference_output: &Array4<f32>,
    tolerance: f32,
    test_name: &str,
) -> Result<(), String> {
    let max_diff =
        kernel_output.iter().zip(reference_output.iter()).map(|(lhs, rhs)| (lhs - rhs).abs()).fold(0.0_f32, f32::max);

    if max_diff >= tolerance {
        return Err(format!(
            "{} output differs from reference by more than {}: max_diff = {}",
            test_name, tolerance, max_diff
        ));
    }

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
    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);
    let kernel = create_single_pass_kernel(&context, head_dim, false, false);
    let kernel_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale)
        .expect("single-pass attention");

    compare_results(&kernel_output, &reference_output, 1e-2, "Single-pass attention").unwrap();
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
    let kernel_output = run_gemm_attention(&kernel, &context, &queries, &keys, &values, None, scale, false)
        .expect("run gemm attention");

    compare_results(&kernel_output, &reference_output, 1e-2, "Gemm attention").unwrap();
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
    let kernel_output = run_gemm_attention(&kernel, &context, &queries, &keys, &values, None, scale, false)
        .expect("run gemm attention f32 head_dim=128");

    compare_results(&kernel_output, &reference_output, 1e-2, "Gemm attention f32 head_dim=128").unwrap();
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

    let reference_output = reference_attention(&queries, &keys, &values, None, scale, is_causal);
    let kernel = create_single_pass_kernel(&context, head_dim, false, is_causal);
    let vector_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale)
        .expect("run vector attention");
    let gemm_kernel = AttentionGemmBlock::new(DataType::F32);
    let matrix_output = run_gemm_attention(&gemm_kernel, &context, &queries, &keys, &values, None, scale, is_causal)
        .expect("run matrix attention");
    let tol_cpu = 5e-2;
    compare_results(&vector_output, &reference_output, tol_cpu, "vector single-pass attention vs CPU").unwrap();
    compare_results(&matrix_output, &reference_output, tol_cpu, "matrix attention vs CPU").unwrap();

    let max_diff_vector_matrix =
        vector_output.iter().zip(matrix_output.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let tol_vector_matrix = 5e-2;
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
    let kernel = create_single_pass_kernel(&context, head_dim, true, false);

    let reference_output = reference_attention(&queries, &keys, &values, Some(&sinks), scale, false);
    let kernel_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, Some(&sinks), scale)
        .expect("single-pass attention with sinks");

    compare_results(&kernel_output, &reference_output, 1e-2, "Single-pass attention with sinks").unwrap();
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
    let kernel = create_single_pass_kernel(&context, head_dim, true, false);

    let reference_output = reference_attention(&queries, &keys, &values, Some(&sinks), scale, false);
    let kernel_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, Some(&sinks), scale)
        .expect("single-pass attention with sinks (long sequence)");

    compare_results(&kernel_output, &reference_output, 5e-2, "Single-pass attention with sinks (long sequence)")
        .unwrap();
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

    let kernel = create_single_pass_kernel(&context, head_dim, false, false);
    let kernel_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale)
        .expect("single-pass attention GQA");

    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);
    compare_results(&kernel_output, &reference_output, 1e-2, "Single-pass attention GQA").unwrap();
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

    let queries_buffer = create_query_allocation(queries, context);
    let keys_buffer = create_attention_cache_allocation(keys, seq_len, context);
    let values_buffer = create_attention_cache_allocation(values, seq_len, context);
    let sinks_buffer = sinks.map(|sinks| create_sinks_allocation(sinks, context));

    let total_blocks_count = 32;
    let partials_size = num_heads * seq_len * total_blocks_count * head_dim;
    let sums_maxs_size = num_heads * seq_len * total_blocks_count;

    let mut partials_buffer = alloc_allocation::<Metal, f32>(context, partials_size);
    let mut sums_buffer = alloc_allocation::<Metal, f32>(context, sums_maxs_size);
    let mut maxs_buffer = alloc_allocation::<Metal, f32>(context, sums_maxs_size);
    let mut output_buffer = alloc_allocation::<Metal, f32>(context, num_heads * seq_len * head_dim);

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
        None::<&Allocation<Metal>>,
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
    submit_encoder(encoder);

    let output_slice: Vec<f32> = allocation_to_vec(&output_buffer);
    let kernel_output = convert_kernel_output(&output_slice, batch_size, num_heads, seq_len, head_dim);

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

    let (kernel_pass1, kernel_pass2) = create_two_pass_kernels(&context, head_dim, is_causal);
    let kernel_output =
        run_two_pass_attention(&kernel_pass1, &kernel_pass2, &context, &queries, &keys, &values, None, scale)
            .expect("two-pass attention");

    compare_results(&kernel_output, &reference_output, 1e-2, "Two-pass attention").unwrap();
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

    let (kernel_pass1, kernel_pass2) = create_two_pass_kernels(&context, head_dim, is_causal);
    let kernel_output =
        run_two_pass_attention(&kernel_pass1, &kernel_pass2, &context, &queries, &keys, &values, None, scale)
            .expect("two-pass attention GQA");

    compare_results(&kernel_output, &reference_output, 1e-2, "Two-pass attention GQA").unwrap();
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

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 123);
    let (kernel_pass1, kernel_pass2) = create_two_pass_kernels(&context, head_dim, is_causal);

    let suffix_start = (seq_len - suffix_length) as usize;
    let queries_suffix = queries.slice(s![.., .., suffix_start.., ..]).to_owned();
    let queries_buffer = create_query_allocation(&queries_suffix, &context);
    let keys_buffer = create_attention_cache_allocation(&keys, seq_len, &context);
    let values_buffer = create_attention_cache_allocation(&values, seq_len, &context);

    let total_blocks_count = 32;
    let partials_size = num_heads * suffix_length * total_blocks_count * head_dim;
    let sums_maxs_size = num_heads * suffix_length * total_blocks_count;

    let mut partials_buffer = alloc_allocation::<Metal, f32>(context.as_ref(), partials_size);
    let mut sums_buffer = alloc_allocation::<Metal, f32>(context.as_ref(), sums_maxs_size);
    let mut maxs_buffer = alloc_allocation::<Metal, f32>(context.as_ref(), sums_maxs_size);
    let mut output_buffer = alloc_allocation::<Metal, f32>(context.as_ref(), num_heads * suffix_length * head_dim);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    let sinks_buffer: Option<Allocation<Metal>> = None;
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
        None::<&Allocation<Metal>>,
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

    let output_slice: Vec<f32> = allocation_to_vec(&output_buffer);
    for &val in output_slice.iter().take(100) {
        assert!(val.is_finite(), "Output contains non-finite values");
    }
}
