use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::trie::TrieNode};

#[kernel(AttentionGemm)]
#[variants(T, f32, f16, bf16)]
#[variants(BK, 16, 32)]
#[variants(BD, 64, 128, 256)]
#[variants(USE_MXU, false)]
#[allow(unused_variables)]
pub fn attention_gemm<T: ArrayElement + Float, const BK: u32, const BD: u32, const USE_MXU: bool>(
    q: *const T,
    k: *const T,
    v: *const T,
    o: *mut T,
    params: crate::backends::common::gpu_types::attention::AttnParams,
    #[optional(is_kv_cache_ring)] ring_params: Option<crate::backends::common::gpu_types::ring::RingParams>,
    #[optional(is_trie)] trie: Option<*const TrieNode>,
    #[optional(is_sliding_window)] sliding_window_size: Option<u32>,
    #[optional(has_sinks)] sinks: Option<*const T>,
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
    unreachable!("CPU AttentionGemm is only generated to satisfy the shared kernel trait")
}
