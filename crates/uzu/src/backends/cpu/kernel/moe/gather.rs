use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeGatherXPerm2D)]
#[variants(T, f32, f16, bf16)]
pub fn moe_gather_x_perm2_d<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] bucketed_ids: *const i32,
    #[allow(unused)] x_perm: *mut T,
    #[allow(unused)] sumk_buf: *const u32,
    #[allow(unused)] d_model: u32,
    #[allow(unused)] t: u32,
    #[allow(unused)] k: u32,
) {
    todo!()
}

#[kernel(MoeGatherXPerm1D)]
#[variants(T, f32, f16, bf16)]
pub fn moe_gather_x_perm1_d<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] bucketed_ids: *const i32,
    #[allow(unused)] x_perm: *mut T,
    #[allow(unused)] sumk_buf: *const u32,
    #[allow(unused)] d_model: u32,
    #[allow(unused)] t: u32,
    #[allow(unused)] k: u32,
) {
    todo!()
}
