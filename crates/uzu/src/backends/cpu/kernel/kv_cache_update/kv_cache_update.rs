use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(KVCacheUpdate)]
#[variants(T, f32, bf16, f16)]
pub fn kv_cache_update<T: ArrayElement + Float>(
    in_place_keys: *mut T,
    in_place_values: *mut T,
    swaps: &[crate::backends::common::gpu_types::kv_cache_update::Swap],
    swap_count: u32,
    num_heads: u32,
    max_sequence_length: u32,
    head_dim: u32,
) {
    let max_sequence_length = max_sequence_length as usize;
    let head_dim = head_dim as usize;

    for head_idx in 0..num_heads as usize {
        for channel_idx in 0..head_dim {
            for i in 0..swap_count as usize {
                let head_offset = head_idx * max_sequence_length * head_dim;
                let source_idx = head_offset + (swaps[i].source as usize) * head_dim + channel_idx;
                let dest_idx = head_offset + (swaps[i].destination as usize) * head_dim + channel_idx;
                unsafe {
                    let keys_src_ptr = in_place_keys.add(source_idx);
                    let mut keys_dst_ptr = in_place_keys.add(dest_idx);
                    keys_src_ptr.swap(keys_dst_ptr);

                    let values_src_ptr = in_place_values.add(source_idx);
                    let mut values_dst_ptr = in_place_values.add(dest_idx);
                    values_src_ptr.swap(values_dst_ptr);
                }
            }
        }
    }
}
