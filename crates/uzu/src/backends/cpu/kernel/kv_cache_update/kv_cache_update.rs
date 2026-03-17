use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(KVCacheUpdate)]
#[variants(T, f32, bf16, f16)]
pub fn kv_cache_update<T: ArrayElement + Float>(
    #[allow(unused)] in_place_keys: *mut T,
    #[allow(unused)] in_place_values: *mut T,
    #[allow(unused)] swaps: &[crate::backends::common::gpu_types::kv_cache_update::Swap],
    #[allow(unused)] swap_count: u32,
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] max_sequence_length: u32,
    #[allow(unused)] head_dim: u32,
) {
    for swap_idx in 0..swap_count as usize {
        let swap = unsafe { &*swaps.as_ptr().add(swap_idx) };
        let src = swap.source as usize;
        let dst = swap.destination as usize;

        for head in 0..num_heads as usize {
            for d in 0..head_dim as usize {
                let src_idx = head * max_sequence_length as usize * head_dim as usize
                    + src * head_dim as usize
                    + d;
                let dst_idx = head * max_sequence_length as usize * head_dim as usize
                    + dst * head_dim as usize
                    + d;

                unsafe {
                    let k_tmp = *in_place_keys.add(src_idx);
                    *in_place_keys.add(src_idx) = *in_place_keys.add(dst_idx);
                    *in_place_keys.add(dst_idx) = k_tmp;

                    let v_tmp = *in_place_values.add(src_idx);
                    *in_place_values.add(src_idx) = *in_place_values.add(dst_idx);
                    *in_place_values.add(dst_idx) = v_tmp;
                };
            }
        }
    }
}
