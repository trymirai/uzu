use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorAddBias)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_bias<T: ArrayElement + Float>(
    #[allow(unused)]
    #[optional(!in_place)]
    input: Option<*const T>,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] num_cols: u32,
    #[allow(unused)] length: u32,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    todo!()
}
