use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Sigmoid)]
#[variants(T, f32, f16, bf16)]
pub fn sigmoid<T: ArrayElement + Float>(
    input: *const T,
    output: *mut T,
    total_elements: u32,
) {
    for i in 0..total_elements as usize {
        unsafe {
            let input_value = (*input.add(i)).to_f32().unwrap();
            let value = 1.0 / (1.0 + (-input_value).exp());
            *output.add(i) = T::from(value).unwrap();
        }
    }
}
