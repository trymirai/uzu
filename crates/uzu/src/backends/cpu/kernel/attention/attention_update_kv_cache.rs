use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AttentionUpdateKVCache)]
#[variants(T, f32, f16, bf16)]
pub fn attention_update_kv_cache<T: ArrayElement + Float>(
    #[allow(unused)] rotated_keys: *const T,
    #[allow(unused)] qkv: *const T,
    #[allow(unused)] key_cache: *mut T,
    #[allow(unused)] value_cache: *mut T,
    #[allow(unused)] num_groups: u32,
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] head_dim: u32,
    #[allow(unused)] suffix_length: u32,
    #[allow(unused)] prefix_segment_length: u32,
    #[allow(unused)] max_sequence_length: u32,
) {
    todo!()
}
