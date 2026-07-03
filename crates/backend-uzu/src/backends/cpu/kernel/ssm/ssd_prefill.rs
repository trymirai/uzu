use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(SSDPrefill64)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_prefill64<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] dt_raw: *const T,
    #[allow(unused)] b: *const T,
    #[allow(unused)] c: *const T,
    #[allow(unused)] d: *const T,
    #[allow(unused)] z: *const T,
    #[allow(unused)] state: *mut T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] group_size: u32,
    #[allow(unused)] state_size: u32,
    #[allow(unused)] x_strides: &[u32; 3],
    #[allow(unused)] dt_strides: &[u32; 2],
    #[allow(unused)] cb_strides: &[u32; 3],
    #[allow(unused)] state_strides: &[u32; 3],
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] head_dim: u32,
) {
    todo!()
}

#[kernel(SSDPrefill)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_prefill<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] dt_raw: *const T,
    #[allow(unused)] b: *const T,
    #[allow(unused)] c: *const T,
    #[allow(unused)] d: *const T,
    #[allow(unused)] z: *const T,
    #[allow(unused)] state: *mut T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] group_size: u32,
    #[allow(unused)] state_size: u32,
    #[allow(unused)] x_strides: &[u32; 3],
    #[allow(unused)] dt_strides: &[u32; 2],
    #[allow(unused)] cb_strides: &[u32; 3],
    #[allow(unused)] state_strides: &[u32; 3],
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] head_dim: u32,
) {
    todo!()
}
