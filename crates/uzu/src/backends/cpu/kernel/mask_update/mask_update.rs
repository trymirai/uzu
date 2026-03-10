use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MaskUpdate)]
#[variants(T, f32, f16, bf16)]
pub fn mask_update<T: ArrayElement + Float>(
    #[allow(unused)] mask: *mut T,
    #[allow(unused)] unmask_col: i32,
    #[allow(unused)] mask_col: i32,
) {
    todo!()
}
