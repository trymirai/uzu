use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(ElementWiseMulStrided)]
#[variants(T, f32, f16, bf16)]
pub fn element_wise_mul_strided<T: ArrayElement + Float>(
    #[allow(unused)] input_a: *const T,
    #[allow(unused)] input_b: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] ple_dim: u32,
    #[allow(unused)] stride: u32,
    #[allow(unused)] layer_offset: u32,
    #[allow(unused)] rows: u32,
) {
    unsafe {
        for row in 0usize..(rows as usize) {
            for col in 0usize..(ple_dim as usize) {
                let a_idx = row * (ple_dim as usize) + col;
                let b_idx = row * (stride as usize) + (layer_offset as usize) + col;
                let a = (*input_a.add(a_idx)).to_f32().unwrap();
                let b = (*input_b.add(b_idx)).to_f32().unwrap();
                *output.add(a_idx) = T::from(a * b).unwrap();
            }
        }
    }
}
