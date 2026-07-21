use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(WeaverNodeCacheWrite)]
#[variants(T, f32, bf16)]
pub fn weaver_node_cache_write<T: ArrayElement + Float>(
    current_qkv: *const T,
    node_qkv: *mut T,
    node_indices: *const u32,
    model_dim: u32,
    total: u32,
) {
    let model_dim = model_dim as usize;
    let qkv_width = 3 * model_dim;
    unsafe {
        for position in 0..total as usize {
            let row = position / (2 * model_dim);
            let offset = position - row * 2 * model_dim;
            *node_qkv.add(*node_indices.add(row) as usize * qkv_width + model_dim + offset) =
                *current_qkv.add(row * qkv_width + model_dim + offset);
        }
    }
}
