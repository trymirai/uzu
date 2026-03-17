use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Conv1dPack)]
#[variants(T, f32, f16, bf16)]
pub fn conv1d_pack<T: ArrayElement + Float>(
    #[allow(unused)] state_in: *const T,
    #[allow(unused)] x: *const T,
    #[allow(unused)] padded: *mut T,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)] row_stride: u32,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] num_channels: u32,
) {
    todo!()
}

#[kernel(Conv1dDecode)]
#[variants(T, f32, f16, bf16)]
pub fn conv1d_decode<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] w: *const T,
    #[allow(unused)]
    #[optional(has_bias)]
    b: Option<*const T>,
    #[allow(unused)]
    #[optional(!state_in_place)]
    state: Option<*const T>,
    #[allow(unused)] x_out: *mut T,
    #[allow(unused)] b_out: *mut T,
    #[allow(unused)] c_out: *mut T,
    #[allow(unused)] next_state: *mut T,
    #[allow(unused)] kernel_size: u32,
    #[allow(unused)] row_stride: u32,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)] num_channels: u32,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] inner_dim: u32,
    #[allow(unused)] proj_dim: u32,
    #[allow(unused)] activation_type: crate::backends::common::gpu_types::activation::ActivationType,
    #[allow(unused)]
    #[specialize]
    has_bias: bool,
    #[allow(unused)]
    #[specialize]
    state_in_place: bool,
) {
    todo!()
}

#[kernel(Conv1dScan)]
#[variants(T, f32, f16, bf16)]
pub fn conv1d_scan<T: ArrayElement + Float>(
    #[allow(unused)] padded: *const T,
    #[allow(unused)] w: *const T,
    #[allow(unused)]
    #[optional(has_bias)]
    b: Option<*const T>,
    #[allow(unused)] x_out: *mut T,
    #[allow(unused)] b_out: *mut T,
    #[allow(unused)] c_out: *mut T,
    #[allow(unused)] state_out: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] kernel_size: u32,
    #[allow(unused)] row_stride: u32,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)] num_channels: u32,
    #[allow(unused)] inner_dim: u32,
    #[allow(unused)] proj_dim: u32,
    #[allow(unused)] activation_type: crate::backends::common::gpu_types::activation::ActivationType,
    #[allow(unused)]
    #[specialize]
    has_bias: bool,
) {
    todo!()
}
