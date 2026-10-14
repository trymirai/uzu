use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::{common::gpu_types::trie::TrieNode, cpu::kernel::attention::mask::should_use_key},
};

#[kernel(AttentionSinglePass)]
#[variants(T, f32, bf16)]
#[variants(HEAD_DIM, 64, 128, 256, 512)]
pub fn attention_single_pass<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    keys: *const T,
    values: *const T,
    #[optional(has_rope)] cosines: Option<*const f32>,
    #[optional(has_rope)] sines: Option<*const f32>,
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
    #[optional(has_sinks)] sinks: Option<*const T>,
    num_heads: u32,
    suffix_length: u32,
    #[optional(has_rope)] rope_dim: Option<u32>,
    #[optional(has_rope)] rope_position_offset: Option<u32>,
    #[specialize] has_sinks: bool,
    #[specialize] is_kv_cache_ring: bool,
    #[specialize] is_causal: bool,
    #[specialize] is_trie: bool,
    #[specialize] is_sliding_window: bool,
    #[specialize] has_rope: bool,
) {
    assert_eq!(ring_params.is_some(), is_kv_cache_ring);
    assert_eq!(sliding_window_size.is_some(), is_sliding_window);
    if has_rope {
        let dim = rope_dim.unwrap();
        assert!(dim > 0 && dim.is_multiple_of(2) && dim <= HEAD_DIM, "invalid RoPE dimension");
    }

    let load = |head: *const T, dim: usize, position: u32| {
        let input = unsafe { *head.add(dim) }.to_f32().unwrap();
        if !has_rope || dim >= rope_dim.unwrap() as usize {
            return input;
        }
        let half = rope_dim.unwrap() as usize / 2;
        let first_half = dim < half;
        let paired_dim = if first_half {
            dim + half
        } else {
            dim - half
        };
        let paired = unsafe { *head.add(paired_dim) }.to_f32().unwrap();
        let signed_paired = if first_half {
            -paired
        } else {
            paired
        };
        let index = position as usize * rope_dim.unwrap() as usize + dim;
        let cosine = unsafe { *cosines.unwrap().add(index) };
        let sine = unsafe { *sines.unwrap().add(index) };
        T::from(input * cosine + signed_paired * sine).unwrap().to_f32().unwrap()
    };

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
            let query_rope_position = if has_rope {
                rope_position_offset.unwrap() + query_position - suffix_position
            } else {
                0
            };
            let mut q = vec![0.0f32; HEAD_DIM as usize];
            let mut o = vec![0.0f32; HEAD_DIM as usize];
            for j in 0..HEAD_DIM as usize {
                q[j] = scale * load(queries, j, query_rope_position);
            }

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp_score = 0.0f32;
            if has_sinks {
                let q_head_idx = head_idx % num_heads;
                max_score = unsafe { *sinks.unwrap().add(q_head_idx as usize) }.to_f32().unwrap();
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
                    let key_rope_position = if !has_rope {
                        0
                    } else if i >= prefix_length {
                        let suffix_idx = i - prefix_length;
                        let position = if is_trie {
                            unsafe { (*trie.unwrap().add(suffix_idx as usize)).height }
                        } else {
                            suffix_idx
                        };
                        rope_position_offset.unwrap() + position
                    } else if let Some(ring) = ring_params {
                        let logical_idx = (prefix_length + i - ring.ring_offset) % prefix_length;
                        rope_position_offset.unwrap() - prefix_length + logical_idx
                    } else {
                        i
                    };

                    // Compute the i-th score
                    let mut score = 0.0f32;
                    for j in 0..HEAD_DIM as usize {
                        score += q[j] * load(keys, j, key_rope_position);
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
