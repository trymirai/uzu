use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorCopy)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_copy<T: ArrayElement + Float>(
    #[allow(unused)] src_buffer: *const T,
    #[allow(unused)] dst_buffer: *mut T,
    #[allow(unused)] length: u32,
) {
    unsafe {
        for i in 0usize..(length as usize) {
            *dst_buffer.offset(i as isize) = *src_buffer.offset(i as isize);
        }
    }
}
