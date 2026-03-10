use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(SplitInProj)]
#[variants(T, f32, f16, bf16)]
pub fn split_in_proj<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] conv_out: *mut T,
    #[allow(unused)] z_out: *mut T,
    #[allow(unused)] dt_out: *mut T,
    #[allow(unused)] z_bias: *const T,
    #[allow(unused)] suffix_length: u32,
    #[allow(unused)] total_dim: u32,
    #[allow(unused)] conv_dim: u32,
    #[allow(unused)] inner_dim: u32,
    #[allow(unused)] num_heads: u32,
) {
    todo!()
}
