use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{array::ArrayElement, backends::common::kernel::kv_layout::Int8KVLayout};

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
) {
    let layout = Int8KVLayout {
        head_dim,
        num_kv_heads,
        padded_capacity,
    };
    for copy in copies.iter().take(copy_count as usize) {
        for head in 0..num_kv_heads {
            for channel in 0..head_dim {
                let source_k = layout.k_code_index(copy.source, head, channel);
                let destination_k = layout.k_code_index(copy.destination, head, channel);
                let source_v = layout.v_code_index(copy.source, head, channel);
                let destination_v = layout.v_code_index(copy.destination, head, channel);
                unsafe {
                    *in_place_k_codes.add(destination_k) = *in_place_k_codes.add(source_k);
                    *in_place_v_codes.add(destination_v) = *in_place_v_codes.add(source_v);
                }
            }
            let source = (head * padded_capacity + copy.source) as usize;
            let destination = (head * padded_capacity + copy.destination) as usize;
            unsafe {
                *in_place_k_scales.add(destination) = *in_place_k_scales.add(source);
                *in_place_v_scales.add(destination) = *in_place_v_scales.add(source);
            }
        }
    }
}
