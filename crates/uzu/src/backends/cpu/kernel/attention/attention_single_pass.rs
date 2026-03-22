use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    ArrayElement,
    backends::{common::gpu_types::trie::TrieNode, cpu::kernel::attention::mask::should_use_key},
};

#[kernel(AttentionSinglePass)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_single_pass<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    keys: *const T,
    values: *const T,
    out: *mut T,
    gqa_factor: u32,
    sequence_length: u32,
    k_head_stride: u32,
    k_seq_stride: u32,
    v_head_stride: u32,
    v_seq_stride: u32,
    #[optional(is_kv_cache_ring)] ring_params: Option<crate::backends::common::gpu_types::ring::RingParams>,
    scale: f32,
    #[optional(is_trie)] trie: Option<*const TrieNode>,
    #[optional(is_sliding_window)] sliding_window_size: Option<u32>,
    #[optional(has_sinks)] sinks: Option<*const f32>,
    num_heads: u32,
    suffix_length: u32,
    #[specialize] has_sinks: bool,
    #[specialize] is_kv_cache_ring: bool,
    #[specialize] is_causal: bool,
    #[specialize] is_trie: bool,
    #[specialize] is_sliding_window: bool,
) {
    let value_dim = HEAD_DIM;

    let prefix_length = sequence_length - suffix_length;
    let suffix_position = if let Some(ring_params) = ring_params {
        ring_params.ring_length
    } else {
        prefix_length
    };

    for head_idx in 0..num_heads {
        for q_seq_idx in 0..suffix_length {
            let kv_head_idx = head_idx / gqa_factor;
            let o_offset = q_seq_idx * num_heads + head_idx;
            let q_offset = head_idx * suffix_length + q_seq_idx;

            let query_position = if is_trie {
                let trie_node = unsafe { &*trie.unwrap().add(q_seq_idx as usize) };
                suffix_position + trie_node.height
            } else {
                suffix_position + q_seq_idx
            };

            let queries: *const T = unsafe { queries.add((q_offset * HEAD_DIM) as usize) };
            let keys: *const T = unsafe { keys.add((kv_head_idx * k_head_stride) as usize) };
            let values: *const T = unsafe { values.add((kv_head_idx * v_head_stride) as usize) };
            let out: *mut T = unsafe { out.add((o_offset * value_dim) as usize) };

            // Read the query and 0 the output accumulator
            let mut q = vec![0.0f32; HEAD_DIM as usize];
            let mut o = vec![0.0f32; HEAD_DIM as usize];
            for j in 0..HEAD_DIM as usize {
                q[j] = scale * unsafe { *queries.add(j) }.to_f32().unwrap();
            }

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp_score = 0.0f32;
            if has_sinks {
                let q_head_idx = head_idx % num_heads;
                max_score = unsafe { *sinks.unwrap().add(q_head_idx as usize) };
                sum_exp_score = 1.0;
            }

            // For each key
            for i in 0..sequence_length {
                if should_use_key(
                    ring_params,
                    trie,
                    sliding_window_size,
                    q_seq_idx,
                    prefix_length,
                    suffix_position,
                    query_position,
                    i,
                    is_causal,
                ) {
                    let keys = unsafe { keys.add((i * k_seq_stride) as usize) };

                    // Compute the i-th score
                    let mut score = 0.0f32;
                    for j in 0..HEAD_DIM as usize {
                        score += q[j] * unsafe { *keys.add(j) }.to_f32().unwrap();
                    }

                    // Update the accumulators
                    let new_max = f32::max(max_score, score);
                    let factor = (max_score - new_max).exp();
                    let exp_score = (score - new_max).exp();

                    max_score = new_max;
                    sum_exp_score = sum_exp_score * factor + exp_score;

                    // Update the output accumulator
                    let values = unsafe { values.add((i * v_seq_stride) as usize) };
                    for j in 0..HEAD_DIM as usize {
                        o[j] = o[j] * factor + exp_score * unsafe { *values.add(j) }.to_f32().unwrap();
                    }
                }
            }

            // Write the output
            for j in 0..HEAD_DIM as usize {
                unsafe {
                    *out.add(j) = T::from(o[j] / sum_exp_score).unwrap();
                }
            }
        }
    }
}
