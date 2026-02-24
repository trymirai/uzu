use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorAddSwap)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_swap<T: ArrayElement + Float>(
    #[allow(unused)] skip_buffer: *mut T,
    #[allow(unused)] main_buffer: *mut T,
    #[allow(unused)] length: u32,
) {
    todo!()
}
