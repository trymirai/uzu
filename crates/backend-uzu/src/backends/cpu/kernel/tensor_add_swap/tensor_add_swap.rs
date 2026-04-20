use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorAddSwap)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_swap<T: ArrayElement + Float>(
    skip_buffer: *mut T,
    main_buffer: *mut T,
    length: u32,
) {
    unsafe {
        for i in 0..length {
            let result = *skip_buffer.add(i as usize) + *main_buffer.add(i as usize);
            *skip_buffer.add(i as usize) = result;
            *main_buffer.add(i as usize) = result;
        }
    }
}
