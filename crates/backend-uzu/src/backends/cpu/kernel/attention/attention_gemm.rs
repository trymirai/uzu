use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    ArrayElement,
    backends::{common::gpu_types::trie::TrieNode, cpu::kernel::attention::mask::should_use_key},
};

#[kernel(AttentionGemm)]
#[variants(T, f32, f16, bf16)]
#[variants(BK, 16, 32)]
#[variants(BD, 64, 128, 256)]
pub fn attention_gemm<T: ArrayElement + Float, const BK: u32, const BD: u32>(
    q: *const T,
    k: *const T,
    v: *const T,
    o: *mut T,
    params: crate::backends::common::gpu_types::attention::AttnParams,
    #[optional(is_kv_cache_ring)] ring_params: Option<crate::backends::common::gpu_types::ring::RingParams>,
    #[optional(is_trie)] trie: Option<*const TrieNode>,
    #[optional(is_sliding_window)] sliding_window_size: Option<u32>,
    #[optional(has_sinks)] sinks: Option<*const f32>,
    num_heads: u32,
    suffix_length: u32,
    #[specialize] align_q: bool,
    #[specialize] align_k: bool,
    #[specialize] is_kv_cache_ring: bool,
    #[specialize] is_causal: bool,
    #[specialize] is_trie: bool,
    #[specialize] is_sliding_window: bool,
    #[specialize] has_sinks: bool,
) {
    let q_len = params.q_len as usize;
    let k_len = params.k_len as usize;
    let head_dim = BD as usize;
    let prefix_length = params.q_off as usize;
    let suffix_position = if is_kv_cache_ring {
        ring_params.unwrap().ring_length as usize
    } else {
        prefix_length
    };

    for head_idx in 0..num_heads as usize {
        let kv_head_idx = head_idx / params.gqa_factor as usize;

        let q_head = unsafe { q.add(head_idx * params.q_strides[1] as usize) };
        let k_head = unsafe { k.add(kv_head_idx * params.k_strides[1] as usize) };
        let v_head = unsafe { v.add(kv_head_idx * params.v_strides[1] as usize) };
        let o_head = unsafe { o.add(head_idx * params.o_strides[1] as usize) };

        for qi in 0..q_len {
            let q_row = unsafe { q_head.add(qi * params.q_strides[2] as usize) };
            let o_row = unsafe { o_head.add(qi * params.o_strides[2] as usize) };

            let query_position = if is_trie {
                let trie_node = unsafe { &*trie.unwrap().add(qi) };
                suffix_position + trie_node.height as usize
            } else {
                suffix_position + qi
            };

            // Read query row and pre-scale
            let mut q_vec = vec![0.0f32; head_dim];
            for j in 0..head_dim {
                q_vec[j] = params.scale * unsafe { *q_row.add(j) }.to_f32().unwrap();
            }

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            let mut o_acc = vec![0.0f32; head_dim];

            // Initialize with attention sinks if present
            if has_sinks {
                max_score = unsafe { *sinks.unwrap().add(head_idx) };
                sum_exp = 1.0;
            }

            // Loop over all key positions
            for ki in 0..k_len {
                if !should_use_key(
                    ring_params,
                    trie,
                    sliding_window_size,
                    qi as u32,
                    prefix_length as u32,
                    suffix_position as u32,
                    query_position as u32,
                    ki as u32,
                    is_causal,
                ) {
                    continue;
                }

                // Compute dot product: score = q . k[ki]
                let k_row = unsafe { k_head.add(ki * params.k_strides[2] as usize) };
                let mut score = 0.0f32;
                for j in 0..head_dim {
                    score += q_vec[j] * unsafe { *k_row.add(j) }.to_f32().unwrap();
                }

                // Online softmax update
                let new_max = f32::max(max_score, score);
                let factor = (max_score - new_max).exp();
                let exp_score = (score - new_max).exp();

                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;

                // Update output accumulator
                let v_row = unsafe { v_head.add(ki * params.v_strides[2] as usize) };
                for j in 0..head_dim {
                    o_acc[j] = o_acc[j] * factor + exp_score * unsafe { *v_row.add(j) }.to_f32().unwrap();
                }
            }

            // Normalize and write output
            let inv_sum = 1.0 / sum_exp;
            for j in 0..head_dim {
                unsafe {
                    *o_row.add(j) = T::from(o_acc[j] * inv_sum).unwrap();
                }
            }
        }
    }
}
