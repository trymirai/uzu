use half::bf16;
use uzu_engine_macros::kernel;

use crate::backends::common::kernel::kv_layout::Int8KVLayout;

#[kernel(KVCacheUpdate)]
pub fn kv_cache_update(
    in_place_k_codes: *mut i8,
    in_place_v_codes: *mut u8,
    in_place_k_scales: *mut bf16,
    in_place_v_scales: *mut bf16,
    copies: &[crate::backends::common::gpu_types::kv_cache_update::Copy],
    copy_count: u32,
    #[specialize] head_dim: u32,
    #[specialize] num_kv_heads: u32,
    #[specialize] use_mxu_layout: bool,
) {
    let layout = Int8KVLayout::new(head_dim, num_kv_heads, use_mxu_layout);
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
            let source = layout.scale_index(copy.source, head);
            let destination = layout.scale_index(copy.destination, head);
            unsafe {
                *in_place_k_scales.add(destination) = *in_place_k_scales.add(source);
                *in_place_v_scales.add(destination) = *in_place_v_scales.add(source);
            }
        }
    }
}
