use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{ArrayElement, backends::common::gpu_types::HadamardTransformOrder};

#[kernel(HadamardTransform)]
#[variants(T, f32, f16, bf16)]
pub fn hadamard_transform_mul<T: ArrayElement + Float>(
    #[allow(unused)] data: *mut T,
    #[allow(unused)] factors: *const i32,
    #[allow(unused)] hidden_dim: u32,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)]
    #[specialize]
    transform_order: HadamardTransformOrder,
) {
    todo!()
}
