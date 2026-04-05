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
    x: *const T,
    w: *const T,
    #[optional(has_bias)] b: Option<*const T>,
    #[optional(!state_in_place)] state: Option<*const T>,
    x_out: *mut T,
    b_out: *mut T,
    c_out: *mut T,
    next_state: *mut T,
    kernel_size: u32,
    row_stride: u32,
    state_stride: u32,
    num_channels: u32,
    suffix_len: u32,
    inner_dim: u32,
    proj_dim: u32,
    activation_type: crate::backends::common::gpu_types::activation_type::ActivationType,
    #[specialize] has_bias: bool,
    #[specialize] state_in_place: bool,
) {
    let kernel_size = kernel_size as usize;
    let row_stride = row_stride as usize;
    let state_stride = state_stride as usize;
    let num_channels = num_channels as usize;
    let suffix_len = suffix_len as usize;
    let inner_dim = inner_dim as usize;
    let proj_dim = proj_dim as usize;

    let state = match state_in_place {
        true => next_state,
        false => state.unwrap(),
    };

    unsafe {
        for token_idx in 0..suffix_len {
            for channel_idx in 0..num_channels {
                let x_idx = token_idx * row_stride + channel_idx;
                let state_offset = channel_idx * state_stride;
                let w_row = w.add(channel_idx * kernel_size);

                let mut acc = 0.0f32;
                if has_bias {
                    acc = (*b.unwrap().add(channel_idx)).to_f32().unwrap();
                }

                let state_taps = kernel_size.saturating_sub(1);
                for tap in 0..state_taps {
                    let state_index = state_offset + tap;
                    let sample = (*state.add(state_index)).to_f32().unwrap();
                    acc += (*w_row.add(tap)).to_f32().unwrap() * sample;
                }

                let current = (*x.add(x_idx)).to_f32().unwrap();
                let current_tap_index = state_taps;
                if current_tap_index < kernel_size {
                    acc += (*w_row.add(current_tap_index)).to_f32().unwrap() * current;
                }

                let activated = activation_type.activate(T::from(acc).unwrap());
                if channel_idx < inner_dim {
                    let dst = token_idx * inner_dim + channel_idx;
                    *x_out.add(dst) = activated;
                } else if channel_idx < inner_dim + proj_dim {
                    let dst = token_idx * proj_dim + (channel_idx - inner_dim);
                    *b_out.add(dst) = activated;
                } else if channel_idx < inner_dim + 2 * proj_dim {
                    let dst = token_idx * proj_dim + (channel_idx - inner_dim - proj_dim);
                    *c_out.add(dst) = activated;
                }
                if state_taps == 0 {
                    continue;
                }

                for tap in 0..state_taps - 1 {
                    let src_index = state_offset + tap + 1;
                    let dst_index = state_offset + tap;
                    *next_state.add(dst_index) = *state.add(src_index);
                }

                let tail_index = state_offset + state_taps - 1;
                *next_state.add(tail_index) = T::from(current).unwrap();
            }
        }
    }
}

#[kernel(Conv1dScan)]
#[variants(T, f32, f16, bf16)]
pub fn conv1d_scan<T: ArrayElement + Float>(
    padded: *const T,
    w: *const T,
    #[optional(has_bias)] b: Option<*const T>,
    x_out: *mut T,
    b_out: *mut T,
    c_out: *mut T,
    state_out: *mut T,
    suffix_len: u32,
    kernel_size: u32,
    row_stride: u32,
    state_stride: u32,
    num_channels: u32,
    inner_dim: u32,
    proj_dim: u32,
    activation_type: crate::backends::common::gpu_types::activation_type::ActivationType,
    #[specialize] has_bias: bool,
) {
    let suffix_len = suffix_len as usize;
    let kernel_size = kernel_size as usize;
    let row_stride = row_stride as usize;
    let state_stride = state_stride as usize;
    let num_channels = num_channels as usize;
    let inner_dim = inner_dim as usize;
    let proj_dim = proj_dim as usize;

    unsafe {
        for token_idx in 0..suffix_len + kernel_size.saturating_sub(1) {
            for channel_idx in 0..num_channels {
                let tap_count = kernel_size.saturating_sub(1);
                let state_offset = channel_idx * state_stride;
                let weight_offset = channel_idx * kernel_size;
                let w_row = w.add(weight_offset);

                if token_idx < suffix_len {
                    let mut acc = 0.0f32;
                    if has_bias {
                        acc = (*b.unwrap().add(channel_idx)).to_f32().unwrap();
                    }

                    for tap in 0..kernel_size {
                        let padded_index = (token_idx + tap) * row_stride + channel_idx;
                        let sample = (*padded.add(padded_index)).to_f32().unwrap();
                        acc += (*w_row.add(tap)).to_f32().unwrap() * sample;
                    }

                    let activated = activation_type.activate(T::from(acc).unwrap());
                    if channel_idx < inner_dim {
                        let dst = token_idx * inner_dim + channel_idx;
                        *x_out.add(dst) = activated;
                    } else if channel_idx < inner_dim + proj_dim {
                        let dst = token_idx * proj_dim + (channel_idx - inner_dim);
                        *b_out.add(dst) = activated;
                    } else if channel_idx < inner_dim + 2 * proj_dim {
                        let dst = token_idx * proj_dim + (channel_idx - inner_dim - proj_dim);
                        *c_out.add(dst) = activated;
                    }
                } else if tap_count > 0 {
                    let tap = token_idx - suffix_len;
                    if tap >= tap_count {
                        return;
                    }

                    let padded_index = token_idx * row_stride + channel_idx;
                    let sample = (*padded.add(padded_index)).to_f32().unwrap();
                    *state_out.add(state_offset + tap) = T::from(sample).unwrap();
                }
            }
        }
    }
}
