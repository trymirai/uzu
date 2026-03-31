use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Conv1dPack)]
#[variants(T, f32, f16, bf16)]
pub fn conv1d_pack<T: ArrayElement + Float>(
    state_in: *const T,
    x: *const T,
    padded: *mut T,
    state_stride: u32,
    row_stride: u32,
    suffix_len: u32,
    num_channels: u32,
) {
    let state_stride = state_stride as usize;
    let row_stride = row_stride as usize;
    let suffix_len = suffix_len as usize;
    let num_channels = num_channels as usize;

    unsafe {
        for channel_idx in 0..num_channels {
            for row_idx in 0..state_stride + suffix_len {
                let padded_index = row_idx * row_stride + channel_idx;
                if row_idx < state_stride {
                    let state_index = channel_idx * state_stride + row_idx;
                    *padded.add(padded_index) = *state_in.add(state_index);
                } else {
                    let token = row_idx - state_stride;
                    let x_index = token * row_stride + channel_idx;
                    *padded.add(padded_index) = *x.add(x_index);
                }
            }
        }
    }
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
    #[allow(unused)] activation_type: crate::backends::common::gpu_types::activation_type::ActivationType,
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
    #[allow(unused)] activation_type: crate::backends::common::gpu_types::activation_type::ActivationType,
    #[allow(unused)]
    #[specialize]
    has_bias: bool,
) {
    todo!()
}
