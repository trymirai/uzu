use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    array::ArrayElement,
    encodable_block::mixer::attention::kv_layout::{Int8KvShape, Int8Layout},
};

#[kernel(KVCacheUpdate)]
#[variants(T, f32, bf16)]
pub fn kv_cache_update<T: ArrayElement + Float>(
    in_place_keys: *mut T,
    in_place_values: *mut T,
    copies: &[crate::backends::common::gpu_types::kv_cache_update::Copy],
    copy_count: u32,
    element_dim: u32,
) {
    let element_dim = element_dim as usize;

    for element_idx in 0..element_dim {
        for i in 0..copy_count as usize {
            let source_idx = (copies[i].source as usize) * element_dim + element_idx;
            let dest_idx = (copies[i].destination as usize) * element_dim + element_idx;
            unsafe {
                *in_place_keys.add(dest_idx) = *in_place_keys.add(source_idx);
                *in_place_values.add(dest_idx) = *in_place_values.add(source_idx);
            }
        }
    }
}

#[kernel(KVCacheUpdateInt8)]
pub fn kv_cache_update_int8(
    in_place_k_codes: *mut i8,
    in_place_v_codes: *mut u8,
    in_place_k_scales: *mut bf16,
    in_place_v_scales: *mut bf16,
    copies: &[crate::backends::common::gpu_types::kv_cache_update::Copy],
    copy_count: u32,
    padded_capacity: u32,
    #[specialize] head_dim: u32,
    #[specialize] num_kv_heads: u32,
    #[specialize] tile_keys: u32,
    #[specialize] tile_dims: u32,
) {
    let layout = Int8Layout::new(
        Int8KvShape {
            head_dim,
            num_kv_heads,
            tile_keys,
            tile_dims,
        },
        padded_capacity,
    );
    for copy in &copies[..copy_count as usize] {
        for head in 0..num_kv_heads {
            for channel in 0..head_dim {
                unsafe {
                    *in_place_k_codes.add(layout.k_code_index(copy.destination, head, channel)) =
                        *in_place_k_codes.add(layout.k_code_index(copy.source, head, channel));
                    *in_place_v_codes.add(layout.v_code_index(copy.destination, head, channel)) =
                        *in_place_v_codes.add(layout.v_code_index(copy.source, head, channel));
                }
            }
            let (source, destination) =
                (layout.per_token_index(copy.source, head), layout.per_token_index(copy.destination, head));
            unsafe {
                *in_place_k_scales.add(destination) = *in_place_k_scales.add(source);
                *in_place_v_scales.add(destination) = *in_place_v_scales.add(source);
            }
        }
    }
}
