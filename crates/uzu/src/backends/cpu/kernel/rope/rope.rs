use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Rope)]
#[variants(T, f32, f16, bf16)]
pub fn rope<T: ArrayElement + Float>(
    #[allow(unused)] qkv: *const T,
    #[allow(unused)] cosines: *const T,
    #[allow(unused)] sines: *const T,
    #[allow(unused)] token_positions: *const i32,
    #[allow(unused)] rotated_queries: *mut T,
    #[allow(unused)] rotated_keys: *mut T,
    #[allow(unused)] head_dim: u32,
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] num_groups: u32,
    #[allow(unused)] suffix_length: u32,
    #[allow(unused)] max_sequence_length: u32,
) {
    todo!()
}
