use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(ShortConvPack)]
#[variants(T, f32, f16, bf16)]
pub fn short_conv_pack<T: ArrayElement + Float>(
    #[allow(unused)] state_in: *const T,
    #[allow(unused)] in_proj: *const T,
    #[allow(unused)] padded: *mut T,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] in_proj_stride: u32,
    #[allow(unused)] model_dim: u32,
) {
    let total_rows = state_stride + suffix_len;
    for row_idx in 0..total_rows {
        for channel_idx in 0..model_dim {
            let padded_offset = (row_idx * model_dim + channel_idx) as usize;
            if row_idx < state_stride {
                let state_idx = (channel_idx * state_stride + row_idx) as usize;
                unsafe { *padded.add(padded_offset) = *state_in.add(state_idx) };
            } else {
                let token = row_idx - state_stride;
                let in_proj_idx = (token * in_proj_stride + channel_idx) as usize;
                unsafe {
                    let pre_gate = (*in_proj.add(in_proj_idx)).to_f32().unwrap();
                    let x_in = (*in_proj.add(in_proj_idx + 2 * model_dim as usize)).to_f32().unwrap();
                    let x = x_in * pre_gate;
                    *padded.add(padded_offset) = T::from(x).unwrap();
                };
            }
        }
    }
}

#[kernel(ShortConvPrefill)]
#[variants(T, f32, f16, bf16)]
pub fn short_conv_prefill<T: ArrayElement + Float>(
    #[allow(unused)] padded: *const T,
    #[allow(unused)] in_proj: *const T,
    #[allow(unused)] w: *const T,
    #[allow(unused)]
    #[optional(has_bias)]
    b: Option<*const T>,
    #[allow(unused)] out: *mut T,
    #[allow(unused)] state_out: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] kernel_size: u32,
    #[allow(unused)] in_proj_stride: u32,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)] model_dim: u32,
    #[allow(unused)]
    #[specialize]
    has_bias: bool,
) {
    let tap_count = if kernel_size > 0 { kernel_size - 1 } else { 0 };

    for channel_idx in 0..model_dim {
        let w_row = unsafe { w.add((channel_idx * kernel_size) as usize) };

        // Compute outputs for each token
        for token_idx in 0..suffix_len {
            let mut acc = if has_bias {
                unsafe { (*b.unwrap().add(channel_idx as usize)).to_f32().unwrap() }
            } else {
                0.0f32
            };

            for tap in 0..kernel_size {
                let padded_row = token_idx + tap;
                let padded_offset = (padded_row * model_dim + channel_idx) as usize;
                unsafe {
                    let sample = (*padded.add(padded_offset)).to_f32().unwrap();
                    acc += (*w_row.add(tap as usize)).to_f32().unwrap() * sample;
                };
            }

            let in_proj_idx = (token_idx * in_proj_stride + channel_idx) as usize;
            let post_conv_gate = unsafe { (*in_proj.add(in_proj_idx + model_dim as usize)).to_f32().unwrap() };
            let gated_output = acc * post_conv_gate;

            let out_idx = (token_idx * model_dim + channel_idx) as usize;
            unsafe { *out.add(out_idx) = T::from(gated_output).unwrap() };
        }

        // Write state
        if tap_count > 0 {
            for tap in 0..tap_count {
                let padded_row = suffix_len + tap;
                let padded_offset = (padded_row * model_dim + channel_idx) as usize;
                let state_idx = (channel_idx * state_stride + tap) as usize;
                unsafe { *state_out.add(state_idx) = *padded.add(padded_offset) };
            }
        }
    }
}

#[kernel(ShortConvDecode)]
#[variants(T, f32, f16, bf16)]
pub fn short_conv_decode<T: ArrayElement + Float>(
    #[allow(unused)] in_proj: *const T,
    #[allow(unused)] w: *const T,
    #[allow(unused)]
    #[optional(has_bias)]
    b: Option<*const T>,
    #[allow(unused)]
    #[optional(!state_in_place)]
    state: Option<*const T>,
    #[allow(unused)] out: *mut T,
    #[allow(unused)] next_state: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] kernel_size: u32,
    #[allow(unused)] in_proj_stride: u32,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)] model_dim: u32,
    #[allow(unused)]
    #[specialize]
    has_bias: bool,
    #[allow(unused)]
    #[specialize]
    state_in_place: bool,
) {
    let state_ptr: *const T = if state_in_place {
        next_state as *const T
    } else {
        state.unwrap()
    };

    let tap_count = if kernel_size > 0 { kernel_size - 1 } else { 0 };

    for token_idx in 0..suffix_len {
        for channel_idx in 0..model_dim {
            let state_offset = (channel_idx * state_stride) as usize;
            let w_row = unsafe { w.add((channel_idx * kernel_size) as usize) };

            let in_proj_idx = (token_idx * in_proj_stride + channel_idx) as usize;
            let pre_conv_gate = unsafe { (*in_proj.add(in_proj_idx)).to_f32().unwrap() };
            let post_conv_gate = unsafe { (*in_proj.add(in_proj_idx + model_dim as usize)).to_f32().unwrap() };
            let x_in = unsafe { (*in_proj.add(in_proj_idx + 2 * model_dim as usize)).to_f32().unwrap() };

            let x = x_in * pre_conv_gate;

            let mut acc = if has_bias {
                unsafe { (*b.unwrap().add(channel_idx as usize)).to_f32().unwrap() }
            } else {
                0.0f32
            };

            for tap in 0..tap_count {
                let sample = unsafe { (*state_ptr.add(state_offset + tap as usize)).to_f32().unwrap() };
                acc += unsafe { (*w_row.add(tap as usize)).to_f32().unwrap() } * sample;
            }

            acc += unsafe { (*w_row.add(tap_count as usize)).to_f32().unwrap() } * x;

            let gated_output = acc * post_conv_gate;

            let out_idx = (token_idx * model_dim + channel_idx) as usize;
            unsafe { *out.add(out_idx) = T::from(gated_output).unwrap() };

            // Update state: shift left and append x
            if tap_count > 0 {
                for tap in 0..(tap_count - 1) {
                    unsafe {
                        *next_state.add(state_offset + tap as usize) =
                            *state_ptr.add(state_offset + (tap + 1) as usize);
                    };
                }
                unsafe {
                    *next_state.add(state_offset + (tap_count - 1) as usize) =
                        T::from(x).unwrap();
                };
            }
        }
    }
}

#[kernel(ShortConvTrie)]
#[variants(T, f32, f16, bf16)]
pub fn short_conv_trie<T: ArrayElement + Float>(
    #[allow(unused)] in_proj: *const T,
    #[allow(unused)] w: *const T,
    #[allow(unused)]
    #[optional(has_bias)]
    b: Option<*const T>,
    #[allow(unused)] base_state: *const T,
    #[allow(unused)] parents: *const i32,
    #[allow(unused)] out: *mut T,
    #[allow(unused)] suffix_state: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] kernel_size: u32,
    #[allow(unused)] in_proj_stride: u32,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)] model_dim: u32,
    #[allow(unused)]
    #[specialize]
    has_bias: bool,
) {
    let tap_count = if kernel_size > 0 { kernel_size - 1 } else { 0 };

    for channel_idx in 0..model_dim {
        let w_row = unsafe { w.add((channel_idx * kernel_size) as usize) };
        let base_state_offset = (channel_idx * state_stride) as usize;

        for node in 0..suffix_len {
            let in_proj_idx = (node * in_proj_stride + channel_idx) as usize;
            let pre_conv_gate = unsafe { (*in_proj.add(in_proj_idx)).to_f32().unwrap() };
            let post_conv_gate = unsafe { (*in_proj.add(in_proj_idx + model_dim as usize)).to_f32().unwrap() };
            let x_in = unsafe { (*in_proj.add(in_proj_idx + 2 * model_dim as usize)).to_f32().unwrap() };
            let x = x_in * pre_conv_gate;

            let parent = unsafe { *parents.add(node as usize) };
            let parent_state: *const T = if parent < 0 {
                unsafe { base_state.add(base_state_offset) }
            } else {
                unsafe {
                    suffix_state.add(
                        (parent as u32 * model_dim + channel_idx) as usize * state_stride as usize,
                    )
                }
            };

            let mut acc = if has_bias {
                unsafe { (*b.unwrap().add(channel_idx as usize)).to_f32().unwrap() }
            } else {
                0.0f32
            };

            for tap in 0..tap_count {
                let sample = unsafe { (*parent_state.add(tap as usize)).to_f32().unwrap() };
                acc += unsafe { (*w_row.add(tap as usize)).to_f32().unwrap() } * sample;
            }

            acc += unsafe { (*w_row.add(tap_count as usize)).to_f32().unwrap() } * x;

            let gated_output = acc * post_conv_gate;
            unsafe {
                *out.add((node * model_dim + channel_idx) as usize) =
                    T::from(gated_output).unwrap();
            };

            // Write post-state for this node
            if tap_count > 0 {
                let dst_state = unsafe {
                    suffix_state.add(
                        (node * model_dim + channel_idx) as usize * state_stride as usize,
                    )
                };
                for tap in 0..(tap_count - 1) {
                    unsafe {
                        *dst_state.add(tap as usize) = *parent_state.add((tap + 1) as usize);
                    };
                }
                unsafe {
                    *dst_state.add((tap_count - 1) as usize) = T::from(x).unwrap();
                };
            }
        }
    }
}
