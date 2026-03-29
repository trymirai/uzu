use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MaskUpdate)]
#[variants(T, f32, f16, bf16)]
pub fn mask_update<T: ArrayElement + Float>(
    mask: *mut T,
    unmask_col: i32,
    mask_col: i32,
) {
    if (unmask_col >= 0) {
        unsafe { *mask.add(unmask_col as usize) = T::zero() };
    }
    if (mask_col >= 0) {
        unsafe { *mask.add(mask_col as usize) = T::neg_infinity() };
    }
}
