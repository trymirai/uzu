use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(KVCacheUpdate)]
#[variants(T, f32, bf16, f16)]
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
