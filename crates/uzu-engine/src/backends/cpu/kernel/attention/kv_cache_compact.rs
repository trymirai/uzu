use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{backends::common::kernel::kv_layout::Int8KVLayout, data_type::DataTypeRepr};

#[kernel(KVCacheCompact)]
#[variants(T, f32, bf16)]
pub fn kv_cache_compact<T: DataTypeRepr + Float>(
    in_place_keys: *mut T,
    in_place_values: *mut T,
    accepted_indices: *const u32,
    suffix_base: u32,
    ring_write_start: u32,
    ring_capacity: u32,
    element_dim: u32,
) {
    let element_dim = element_dim as usize;
    let accepted_len = unsafe { *accepted_indices } as usize;

    for element_idx in 0..element_dim {
        for rank in 0..accepted_len {
            let source = suffix_base as usize + unsafe { *accepted_indices.add(accepted_len - rank) } as usize;
            let destination = if ring_capacity == 0 {
                suffix_base as usize + rank
            } else {
                if rank as u32 + ring_capacity < accepted_len as u32 {
                    continue;
                }
                (ring_write_start + rank as u32) as usize % ring_capacity as usize
            };
            if source == destination {
                continue;
            }
            unsafe {
                *in_place_keys.add(destination * element_dim + element_idx) =
                    *in_place_keys.add(source * element_dim + element_idx);
                *in_place_values.add(destination * element_dim + element_idx) =
                    *in_place_values.add(source * element_dim + element_idx);
            }
        }
    }
}

#[kernel(KVCacheCompactInt8)]
pub fn kv_cache_compact_int8(
    in_place_k_codes: *mut i8,
    in_place_v_codes: *mut u8,
    in_place_k_scales: *mut bf16,
    in_place_v_scales: *mut bf16,
    accepted_indices: *const u32,
    suffix_base: u32,
    ring_write_start: u32,
    ring_capacity: u32,
    #[specialize] head_dim: u32,
    #[specialize] num_kv_heads: u32,
) {
    let layout = Int8KVLayout::new(head_dim, num_kv_heads);
    let accepted_len = unsafe { *accepted_indices } as usize;

    for head in 0..num_kv_heads as usize {
        for channel in 0..head_dim as usize {
            for rank in 0..accepted_len {
                let source = suffix_base as usize + unsafe { *accepted_indices.add(accepted_len - rank) } as usize;
                let destination = if ring_capacity == 0 {
                    suffix_base as usize + rank
                } else {
                    if rank as u32 + ring_capacity < accepted_len as u32 {
                        continue;
                    }
                    (ring_write_start + rank as u32) as usize % ring_capacity as usize
                };
                if source == destination {
                    continue;
                }
                unsafe {
                    *in_place_k_codes.add(layout.k_code_index(destination as u32, head as u32, channel as u32)) =
                        *in_place_k_codes.add(layout.k_code_index(source as u32, head as u32, channel as u32));
                    *in_place_v_codes.add(layout.v_code_index(destination as u32, head as u32, channel as u32)) =
                        *in_place_v_codes.add(layout.v_code_index(source as u32, head as u32, channel as u32));
                }
            }
        }
        for rank in 0..accepted_len {
            let source = suffix_base as usize + unsafe { *accepted_indices.add(accepted_len - rank) } as usize;
            let destination = if ring_capacity == 0 {
                suffix_base as usize + rank
            } else {
                if rank as u32 + ring_capacity < accepted_len as u32 {
                    continue;
                }
                (ring_write_start + rank as u32) as usize % ring_capacity as usize
            };
            if source == destination {
                continue;
            }
            unsafe {
                *in_place_k_scales.add(layout.scale_index(destination as u32, head as u32)) =
                    *in_place_k_scales.add(layout.scale_index(source as u32, head as u32));
                *in_place_v_scales.add(layout.scale_index(destination as u32, head as u32)) =
                    *in_place_v_scales.add(layout.scale_index(source as u32, head as u32));
            }
        }
    }
}
