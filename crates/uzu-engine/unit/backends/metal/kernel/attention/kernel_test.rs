use half::bf16;
use ndarray::{Array4, s};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            gpu_types::trie::TrieNode as GpuTrieNode,
            kernel::{AttentionArguments, AttentionKernel},
        },
        cpu::Cpu,
        metal::Metal,
    },
    encodable_block::{mixer::attention::AttentionStateType, sampling::PRng},
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, submit_encoder},
    },
    trie::TrieNode,
};

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
    kernel: &super::single_pass::AttentionSinglePass,
    context: &<Metal as Backend>::Context,
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    sinks: Option<&[f32]>,
    _scale: f32,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let (_batch_size, _num_kv_heads, _seq_len, _head_dim) = keys.dim();

    let query_buffer = create_query_allocation(queries, context);
    let key_cache_buffer = create_attention_cache_allocation(keys, seq_len, context);
    let value_cache_buffer = create_attention_cache_allocation(values, seq_len, context);
    let sinks_buffer = sinks.map(|sinks| create_sinks_allocation(sinks, context));
    let mut encoder = Encoder::new(context).expect("Failed to create encoder");
    let pooled_output = kernel.encode(
        AttentionArguments {
            queries: &query_buffer,
            keys: &key_cache_buffer,
            values: &value_cache_buffer,
            suffix_length: seq_len as u32,
            trie: None,
            sinks: sinks_buffer.as_ref(),
            state_type: &AttentionStateType::Full {
                length: 0,
            },
        },
        &mut encoder,
    )?;
    let mut output_buffer = alloc_allocation::<Metal, f32>(context, num_heads * seq_len * head_dim);
    encoder.encode_copy(&pooled_output, .., &mut output_buffer, ..);
    drop(pooled_output);
    submit_encoder(encoder);

    let output_slice: Vec<f32> = allocation_to_vec(&output_buffer);
    let kernel_output = convert_kernel_output(&output_slice, batch_size, num_heads, seq_len, head_dim);

    Ok(kernel_output)
}

fn create_single_pass_kernel(
    head_dim: usize,
    num_q_heads: usize,
    num_groups: usize,
    has_sinks: bool,
    is_causal: bool,
) -> super::single_pass::AttentionSinglePass {
    let mut config = super::default_attention_config::<f32>(head_dim, num_q_heads, num_groups, is_causal);
    config.has_sinks = has_sinks;
    config.scale = None;
    super::single_pass::AttentionSinglePass::new(&config)
}

fn create_two_pass_kernel(
    head_dim: usize,
    num_q_heads: usize,
    num_groups: usize,
    is_causal: bool,
) -> super::two_pass::AttentionTwoPass {
    let config = super::default_attention_config::<f32>(head_dim, num_q_heads, num_groups, is_causal);
    super::two_pass::AttentionTwoPass::new(&config)
}

fn run_gemm_attention(
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

    let mut config = super::default_attention_config::<f32>(head_dim, num_heads, num_kv_heads, is_causal);
    config.has_sinks = sinks.is_some();
    config.scale = Some(scale);
    let kernel = super::gemm::AttentionGemm::new(&config);

    let query_allocation = create_query_allocation(queries, context);
    let key_allocation = create_attention_cache_allocation(keys, seq_len, context);
    let value_allocation = create_attention_cache_allocation(values, seq_len, context);

    let sinks_allocation = sinks.map(|sinks| create_sinks_allocation(sinks, context));
    let state_type = AttentionStateType::Full {
        length: 0,
    };

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    let args = AttentionArguments {
        queries: &query_allocation,
        keys: &key_allocation,
        values: &value_allocation,
        suffix_length: seq_len as u32,
        trie: None,
        sinks: sinks_allocation.as_ref(),
        state_type: &state_type,
    };

    let pooled_output = kernel.encode(args, &mut encoder)?;
    let mut output_allocation = alloc_allocation::<Metal, f32>(context, num_heads * seq_len * head_dim);
    encoder.encode_copy(&pooled_output, .., &mut output_allocation, ..);
    let completed = encoder.end_encoding().submit().wait_until_completed()?;
    drop(pooled_output);
    drop(completed);

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

#[uzu_test]
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
    let kernel = create_single_pass_kernel(head_dim, num_heads, num_kv_heads, false, false);
    let kernel_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale)
        .expect("single-pass attention");

    compare_results(&kernel_output, &reference_output, 1e-2, "Single-pass attention").unwrap();
}

#[uzu_test]
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
    let kernel = create_single_pass_kernel(head_dim, num_heads, num_kv_heads, false, is_causal);
    let vector_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale)
        .expect("run vector attention");
    let matrix_output =
        run_gemm_attention(&context, &queries, &keys, &values, None, scale, is_causal).expect("run matrix attention");
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

#[uzu_test]
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
    let kernel = create_single_pass_kernel(head_dim, num_heads, num_kv_heads, true, false);

    let reference_output = reference_attention(&queries, &keys, &values, Some(&sinks), scale, false);
    let kernel_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, Some(&sinks), scale)
        .expect("single-pass attention with sinks");

    compare_results(&kernel_output, &reference_output, 1e-2, "Single-pass attention with sinks").unwrap();
}

#[uzu_test]
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
    let kernel = create_single_pass_kernel(head_dim, num_heads, num_kv_heads, true, false);

    let reference_output = reference_attention(&queries, &keys, &values, Some(&sinks), scale, false);
    let kernel_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, Some(&sinks), scale)
        .expect("single-pass attention with sinks (long sequence)");

    compare_results(&kernel_output, &reference_output, 5e-2, "Single-pass attention with sinks (long sequence)")
        .unwrap();
}

#[uzu_test]
fn test_single_pass_attention_gqa() {
    let context = <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context");

    let batch_size = 1;
    let num_heads = 8;
    let num_kv_heads = 2;
    let seq_len = 8;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (queries, keys, values) = create_test_data(batch_size, num_heads, num_kv_heads, seq_len, head_dim, 42);

    let kernel = create_single_pass_kernel(head_dim, num_heads, num_kv_heads, false, false);
    let kernel_output = run_single_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale)
        .expect("single-pass attention GQA");

    let reference_output = reference_attention(&queries, &keys, &values, None, scale, false);
    compare_results(&kernel_output, &reference_output, 1e-2, "Single-pass attention GQA").unwrap();
}

fn run_two_pass_attention(
    kernel: &super::two_pass::AttentionTwoPass,
    context: &<Metal as Backend>::Context,
    queries: &Array4<f32>,
    keys: &Array4<f32>,
    values: &Array4<f32>,
    sinks: Option<&[f32]>,
    _scale: f32,
) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let (batch_size, num_heads, seq_len, head_dim) = queries.dim();
    let queries_buffer = create_query_allocation(queries, context);
    let keys_buffer = create_attention_cache_allocation(keys, seq_len, context);
    let values_buffer = create_attention_cache_allocation(values, seq_len, context);
    let sinks_buffer = sinks.map(|sinks| create_sinks_allocation(sinks, context));
    let state_type = AttentionStateType::Full {
        length: 0,
    };

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    let pooled_output = kernel.encode(
        AttentionArguments {
            queries: &queries_buffer,
            keys: &keys_buffer,
            values: &values_buffer,
            suffix_length: seq_len as u32,
            trie: None,
            sinks: sinks_buffer.as_ref(),
            state_type: &state_type,
        },
        &mut encoder,
    )?;
    let mut output_buffer = alloc_allocation::<Metal, f32>(context, num_heads * seq_len * head_dim);
    encoder.encode_copy(&pooled_output, .., &mut output_buffer, ..);
    drop(pooled_output);
    submit_encoder(encoder);

    let output_slice: Vec<f32> = allocation_to_vec(&output_buffer);
    let kernel_output = convert_kernel_output(&output_slice, batch_size, num_heads, seq_len, head_dim);

    Ok(kernel_output)
}

#[uzu_test]
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

    let kernel = create_two_pass_kernel(head_dim, num_heads, num_kv_heads, is_causal);
    let kernel_output =
        run_two_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale).expect("two-pass attention");

    compare_results(&kernel_output, &reference_output, 1e-2, "Two-pass attention").unwrap();
}

#[uzu_test]
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

    let kernel = create_two_pass_kernel(head_dim, num_heads, num_kv_heads, is_causal);
    let kernel_output = run_two_pass_attention(&kernel, &context, &queries, &keys, &values, None, scale)
        .expect("two-pass attention GQA");

    compare_results(&kernel_output, &reference_output, 1e-2, "Two-pass attention GQA").unwrap();
}

type AttentionShape = (usize, usize, usize, usize, usize, bool);

fn attention_data(shape: AttentionShape) -> (Vec<bf16>, Vec<bf16>) {
    let (head_dim, _, num_groups, suffix_length, prefix_length, _) = shape;
    let row = num_groups * head_dim;
    let count = (prefix_length + suffix_length) * row;
    (fill_attention(count, 1.0), fill_attention(count, 2.0))
}

fn fill_attention(
    count: usize,
    phase: f32,
) -> Vec<bf16> {
    (0..count).map(|i| bf16::from_f32(((i as f32) * 0.017 + phase).sin() * 0.5)).collect()
}

fn run_attention<B: Backend>(
    context: &B::Context,
    shape: AttentionShape,
    trie_nodes: Option<&[GpuTrieNode]>,
) -> Vec<bf16> {
    let (keys, values) = attention_data(shape);
    let (_, _, _, _, prefix_length, _) = shape;
    let (head_dim, num_q_heads, num_groups, _, _, is_causal) = shape;
    let config = super::default_attention_config::<bf16>(head_dim, num_q_heads, num_groups, is_causal);
    let kernel = <B::Kernels as Kernels>::AttentionKernel::new(context, config).expect("attention kernel");
    run_attention_with_kernel::<B>(
        context,
        &kernel,
        shape,
        &keys,
        &values,
        &AttentionStateType::Full {
            length: prefix_length as u32,
        },
        trie_nodes,
    )
}

fn run_attention_with_kernel<B: Backend>(
    context: &B::Context,
    kernel: &<B::Kernels as Kernels>::AttentionKernel,
    shape: AttentionShape,
    keys_data: &[bf16],
    values_data: &[bf16],
    state_type: &AttentionStateType,
    trie_nodes: Option<&[GpuTrieNode]>,
) -> Vec<bf16> {
    let (head_dim, num_q_heads, _, suffix_length, _, _) = shape;
    let queries =
        alloc_allocation_with_data::<B, bf16>(context, &fill_attention(num_q_heads * suffix_length * head_dim, 0.5));
    let keys = alloc_allocation_with_data::<B, bf16>(context, keys_data);
    let values = alloc_allocation_with_data::<B, bf16>(context, values_data);
    let trie = trie_nodes.map(|nodes| {
        let words: Vec<u32> = nodes.iter().flat_map(|node| [node.trie_start, node.trie_end, node.height]).collect();
        alloc_allocation_with_data::<B, u32>(context, &words)
    });
    let arguments = AttentionArguments {
        queries: &queries,
        keys: &keys,
        values: &values,
        suffix_length: suffix_length as u32,
        trie: trie.as_ref(),
        sinks: None,
        state_type,
    };
    let mut encoder = Encoder::<B>::new(context).expect("encoder");
    let pooled = kernel.encode(arguments, &mut encoder).expect("encode");
    let mut output = alloc_allocation::<B, bf16>(context, suffix_length * num_q_heads * head_dim);
    encoder.encode_copy(&pooled, .., &mut output, ..);
    let completed = encoder.end_encoding().submit().wait_until_completed().expect("submit");
    drop(pooled);
    drop(completed);
    allocation_to_vec::<B, bf16>(&output)
}

#[uzu_test]
fn attention_kernel_matches_cpu() {
    let trie: Vec<GpuTrieNode> =
        TrieNode::flat(0, &[0, 1, 2, 3, 4], &PRng::new(0)).linearize().token_subtrie_ranges().collect();
    let cpu_context = <Cpu as Backend>::Context::new().expect("CPU attention context");
    let metal_context = <Metal as Backend>::Context::new().expect("Metal attention context");
    for &(head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal, use_trie) in &[
        (512, 8, 8, 9, 0, false, false),
        (128, 8, 2, 16, 1024, false, false),
        (256, 6, 1, 32, 1024, true, false),
        (256, 6, 1, 31, 1024, true, true),
        (512, 8, 8, 1, 0, false, false),
        (512, 8, 8, 1, 1024, false, false),
        (64, 8, 8, 9, 0, false, false),
    ] {
        let nodes = use_trie.then_some(trie.as_slice());
        let shape = (head_dim, num_q_heads, num_groups, suffix_length, prefix_length, causal);
        let expected = run_attention::<Cpu>(cpu_context.as_ref(), shape, nodes);
        let actual = run_attention::<Metal>(metal_context.as_ref(), shape, nodes);
        let label = format!("attention kernel D{head_dim} S{suffix_length} L{prefix_length} causal={causal}");
        assert_eq_float::<bf16>(&expected, &actual, 1e-2, &label);
    }
}

#[uzu_test]
fn attention_kernel_reuses_instance_for_flat_and_trie() {
    let shape = (256, 6, 1, 5, 1024, true);
    let trie: Vec<GpuTrieNode> =
        TrieNode::flat(0, &[0, 1, 2, 3, 4], &PRng::new(0)).linearize().token_subtrie_ranges().collect();
    let context = <Metal as Backend>::Context::new().expect("Metal attention context");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(
        context.as_ref(),
        super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5),
    )
    .expect("attention kernel");
    let (keys, values) = attention_data(shape);
    let state = AttentionStateType::Full {
        length: 1024,
    };
    let flat = run_attention_with_kernel::<Metal>(context.as_ref(), &kernel, shape, &keys, &values, &state, None);
    let trie_output =
        run_attention_with_kernel::<Metal>(context.as_ref(), &kernel, shape, &keys, &values, &state, Some(&trie));
    let cpu_context = <Cpu as Backend>::Context::new().expect("CPU attention context");
    let cpu_kernel = <<Cpu as Backend>::Kernels as Kernels>::AttentionKernel::new(
        cpu_context.as_ref(),
        super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5),
    )
    .expect("CPU attention kernel");
    let expected_flat =
        run_attention_with_kernel::<Cpu>(cpu_context.as_ref(), &cpu_kernel, shape, &keys, &values, &state, None);
    let expected_trie =
        run_attention_with_kernel::<Cpu>(cpu_context.as_ref(), &cpu_kernel, shape, &keys, &values, &state, Some(&trie));
    assert_eq_float::<bf16>(&expected_flat, &flat, 1e-2, "flat mask specialization");
    assert_eq_float::<bf16>(&expected_trie, &trie_output, 1e-2, "trie mask specialization");
}

#[uzu_test]
fn attention_kernel_ring_matches_full_on_wrap() {
    const HEAD_DIM: usize = 64;
    const NUM_HEADS: usize = 8;
    const CAPACITY: usize = 8;
    const CONTEXT: usize = 12;
    const SUFFIX: usize = 2;
    let shape = (HEAD_DIM, NUM_HEADS, NUM_HEADS, SUFFIX, CONTEXT, true);
    let row = NUM_HEADS * HEAD_DIM;
    let (full_keys, full_values) = attention_data(shape);
    let mut ring_keys = vec![bf16::ZERO; (CAPACITY + SUFFIX) * row];
    let mut ring_values = vec![bf16::ZERO; (CAPACITY + SUFFIX) * row];
    let ring_offset = (CONTEXT - CAPACITY) % CAPACITY;
    for index in 0..CAPACITY {
        let source = CONTEXT - CAPACITY + index;
        let destination = (ring_offset + index) % CAPACITY;
        ring_keys[destination * row..(destination + 1) * row]
            .copy_from_slice(&full_keys[source * row..(source + 1) * row]);
        ring_values[destination * row..(destination + 1) * row]
            .copy_from_slice(&full_values[source * row..(source + 1) * row]);
    }
    for index in 0..SUFFIX {
        let source = CONTEXT + index;
        let destination = CAPACITY + index;
        ring_keys[destination * row..(destination + 1) * row]
            .copy_from_slice(&full_keys[source * row..(source + 1) * row]);
        ring_values[destination * row..(destination + 1) * row]
            .copy_from_slice(&full_values[source * row..(source + 1) * row]);
    }

    let context = <Metal as Backend>::Context::new().expect("Metal attention context");
    let full_kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(context.as_ref(), {
        let mut config = super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5);
        config.sliding_window_size = Some(CAPACITY as u32);
        config
    })
    .expect("full attention kernel");
    let ring_kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(context.as_ref(), {
        let mut config = super::default_attention_config::<bf16>(shape.0, shape.1, shape.2, shape.5);
        config.is_kv_cache_ring = true;
        config.sliding_window_size = Some(CAPACITY as u32);
        config
    })
    .expect("ring attention kernel");
    let full = run_attention_with_kernel::<Metal>(
        context.as_ref(),
        &full_kernel,
        shape,
        &full_keys,
        &full_values,
        &AttentionStateType::Full {
            length: CONTEXT as u32,
        },
        None,
    );
    let ring = run_attention_with_kernel::<Metal>(
        context.as_ref(),
        &ring_kernel,
        shape,
        &ring_keys,
        &ring_values,
        &AttentionStateType::Ring {
            offset: ring_offset as u32,
            length: CAPACITY as u32,
            max_length: CAPACITY as u32,
        },
        None,
    );
    assert_eq_float::<bf16>(&full, &ring, 1e-2, "wrapped ring attention");
}
