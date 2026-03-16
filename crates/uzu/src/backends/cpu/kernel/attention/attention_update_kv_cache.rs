use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AttentionUpdateKVCache)]
#[variants(T, f32, f16, bf16)]
pub fn attention_update_kv_cache<T: ArrayElement + Float>(
    #[optional(!keys_in_place)] rotated_keys: Option<*const T>,
    qkv: *const T,
    key_cache: *mut T,
    value_cache: *mut T,
    num_groups: u32,
    num_heads: u32,
    head_dim: u32,
    total_heads: u32,
    suffix_length: u32,
    prefix_segment_length: u32,
    max_sequence_length: u32,
    #[specialize] keys_in_place: bool,
) {
    let rotated_keys: *const T = match keys_in_place {
        true => key_cache,
        false => rotated_keys.unwrap(),
    };

    for group_index in 0..num_groups {
        for token_index in 0..suffix_length {
            for dim_index in 0..head_dim {
                let cache_token_index = prefix_segment_length + token_index;
                let key_cache_offset = (group_index * max_sequence_length + cache_token_index) * head_dim + dim_index;
                let rotated_key_offset = (group_index * suffix_length + token_index) * head_dim + dim_index;
                unsafe {
                    *key_cache.add(key_cache_offset as usize) = *rotated_keys.add(rotated_key_offset as usize);
                }

                let qkv_stride = total_heads * head_dim;
                let value_base_offset = (num_heads + num_groups) * head_dim;
                let value_offset = token_index * qkv_stride + value_base_offset + group_index * head_dim + dim_index;
                let value_cache_offset = (group_index * max_sequence_length + cache_token_index) * head_dim + dim_index;
                unsafe {
                    *value_cache.add(value_cache_offset as usize) = *qkv.add(value_offset as usize);
                }
            }
        }
    }
}
