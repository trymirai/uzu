use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(SSDUpdate)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_update<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] dt_raw: *const T,
    #[allow(unused)] b: *const T,
    #[allow(unused)] c: *const T,
    #[allow(unused)] d: *const T,
    #[allow(unused)] z: *const T,
    #[allow(unused)] state: *const T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] next_state: *mut T,
    #[allow(unused)] group_size: u32,
    #[allow(unused)] state_size: u32,
    #[allow(unused)] x_strides: &[u32],
    #[allow(unused)] dt_strides: &[u32],
    #[allow(unused)] cb_strides: &[u32],
    #[allow(unused)] state_strides: &[u32],
    #[allow(unused)] b_size: u32,
    #[allow(unused)] h_size: u32,
    #[allow(unused)] dh_size: u32,
) {
    todo!()
}
