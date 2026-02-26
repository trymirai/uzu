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
    todo!()
}
