use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorAddScale)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_scale<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] num_cols: u32,
    #[allow(unused)] length: u32,
    #[allow(unused)] scale: f32,
) {
    todo!()
}
