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
    let head_dim = head_dim as usize;
    let num_heads = num_heads as usize;
    let _ = max_sequence_length;

    // Token-major layout: [max_sequence_length, num_heads, head_dim]
    // Stride between tokens: num_heads * head_dim; stride between heads: head_dim.
    for head_idx in 0..num_heads {
        for channel_idx in 0..head_dim {
            for i in 0..swap_count as usize {
                let head_offset = head_idx * head_dim;
                let source_idx = (swaps[i].source as usize) * num_heads * head_dim + head_offset + channel_idx;
                let dest_idx = (swaps[i].destination as usize) * num_heads * head_dim + head_offset + channel_idx;
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
