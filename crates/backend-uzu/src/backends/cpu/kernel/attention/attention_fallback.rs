use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::{common::gpu_types::trie::TrieNode, cpu::kernel::attention::mask::should_use_key},
};

#[kernel(AttentionFallbackScatterScores)]
#[variants(T, f32, f16, bf16)]
pub fn attention_fallback_scatter_scores<T: ArrayElement + Float>(
    group_scores: *const T,
    scores: *mut T,
    #[optional(is_kv_cache_ring)] ring_params: Option<crate::backends::common::gpu_types::ring::RingParams>,
    #[optional(is_trie)] trie: Option<*const TrieNode>,
    #[optional(is_sliding_window)] sliding_window_size: Option<u32>,
    group_index: u32,
    gqa_factor: u32,
    sequence_length: u32,
    suffix_length: u32,
    total_elements: u32,
    #[specialize] is_kv_cache_ring: bool,
    #[specialize] is_causal: bool,
    #[specialize] is_trie: bool,
    #[specialize] is_sliding_window: bool,
) {
    assert_eq!(ring_params.is_some(), is_kv_cache_ring);
    assert_eq!(sliding_window_size.is_some(), is_sliding_window);
    let expected_total_elements = gqa_factor
        .checked_mul(suffix_length)
        .and_then(|elements| elements.checked_mul(sequence_length))
        .expect("scatter scores element count overflow");
    assert_eq!(total_elements, expected_total_elements);

    let prefix_length = sequence_length - suffix_length;
    let suffix_position = ring_params.map(|params| params.ring_length).unwrap_or(prefix_length);

    for head_in_group in 0..gqa_factor {
        let head_index = group_index * gqa_factor + head_in_group;
        for query_index in 0..suffix_length {
            let query_position = if is_trie {
                let trie_node = unsafe { &*trie.unwrap().add(query_index as usize) };
                suffix_position + trie_node.height
            } else {
                suffix_position + query_index
            };

            for sequence_index in 0..sequence_length {
                let source_index =
                    ((head_in_group * suffix_length + query_index) * sequence_length + sequence_index) as usize;
                let destination_index =
                    ((head_index * suffix_length + query_index) * sequence_length + sequence_index) as usize;
                let use_key = should_use_key(
                    ring_params,
                    trie,
                    sliding_window_size,
                    query_index,
                    prefix_length,
                    suffix_position,
                    query_position,
                    sequence_index,
                    is_causal,
                );
                unsafe {
                    *scores.add(destination_index) = if use_key {
                        *group_scores.add(source_index)
                    } else {
                        T::from(f32::NEG_INFINITY).unwrap()
                    };
                }
            }
        }
    }
}

#[kernel(AttentionFallbackScatterValues)]
#[variants(T, f32, f16, bf16)]
pub fn attention_fallback_scatter_values<T: ArrayElement + Float>(
    group_output: *const T,
    out: *mut T,
    group_index: u32,
    gqa_factor: u32,
    suffix_length: u32,
    num_heads: u32,
    head_dim: u32,
    total_elements: u32,
) {
    let expected_total_elements = gqa_factor
        .checked_mul(suffix_length)
        .and_then(|elements| elements.checked_mul(head_dim))
        .expect("scatter values element count overflow");
    assert_eq!(total_elements, expected_total_elements);

    for head_in_group in 0..gqa_factor as usize {
        for query_index in 0..suffix_length as usize {
            for dim_index in 0..head_dim as usize {
                let source_index =
                    (head_in_group * suffix_length as usize + query_index) * head_dim as usize + dim_index;
                let head_index = group_index as usize * gqa_factor as usize + head_in_group;
                let destination_index = (query_index * num_heads as usize + head_index) * head_dim as usize + dim_index;
                unsafe {
                    *out.add(destination_index) = *group_output.add(source_index);
                }
            }
        }
    }
}
