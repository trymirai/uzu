use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(HadamardTransformMul)]
#[variants(T, f32, f16, bf16)]
pub fn hadamard_transform_mul<T: ArrayElement + Float>(
    #[allow(unused)] data: *mut T,
    #[allow(unused)] factors: *const T,
    #[allow(unused)] total_blocks: u32,
    #[allow(unused)] channel_count: u32,
) {
    todo!()
}
