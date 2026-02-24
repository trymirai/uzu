use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Sigmoid)]
#[variants(T, f32, f16, bf16)]
pub fn sigmoid<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] total_elements: u32,
) {
    todo!()
}
