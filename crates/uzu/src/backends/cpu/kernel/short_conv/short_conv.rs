use dsl::kernel;
use half::{bf16, f16};
use num_traits::{Float, ToPrimitive};

use crate::ArrayElement;

#[kernel(ShortConvPack)]
#[variants(T, f32, f16, bf16)]
pub fn short_conv_pack<T: ArrayElement + Float>(
    state_in: *const T,
    in_proj: *const T,
    padded: *mut T,
    state_stride: u32,
    suffix_len: u32,
    in_proj_stride: u32,
    model_dim: u32,
) {
    let state_stride = state_stride as usize;
    let suffix_len = suffix_len as usize;
    let in_proj_stride = in_proj_stride as usize;
    let model_dim = model_dim as usize;

    unsafe {
        for channel_idx in 0..model_dim {
            for row_idx in 0..state_stride + suffix_len {
                let padded_offset = row_idx * model_dim + channel_idx;
                if row_idx < state_stride {
                    let state_idx = channel_idx * state_stride + row_idx;
                    *padded.add(padded_offset) = *state_in.add(state_idx);
                } else {
                    let token = row_idx - state_stride;
                    let in_proj_idx = token * in_proj_stride + channel_idx;

                    let pre_gate = (*in_proj.add(in_proj_idx)).to_f32().unwrap();
                    let x_in = (*in_proj.add(in_proj_idx + 2 * model_dim)).to_f32().unwrap();
                    let x = x_in * pre_gate;
                    *padded.add(padded_offset) = T::from(x).unwrap();
                }
            }
        }
    }
}

#[kernel(ShortConvPrefill)]
#[variants(T, f32, f16, bf16)]
pub fn short_conv_prefill<T: ArrayElement + Float>(
    padded: *const T,
    in_proj: *const T,
    w: *const T,
    #[optional(has_bias)] b: Option<*const T>,
    out: *mut T,
    state_out: *mut T,
    suffix_len: u32,
    kernel_size: u32,
    in_proj_stride: u32,
    state_stride: u32,
    model_dim: u32,
    #[specialize] has_bias: bool,
) {
    let suffix_len = suffix_len as usize;
    let kernel_size = kernel_size as usize;
    let in_proj_stride = in_proj_stride as usize;
    let state_stride = state_stride as usize;
    let model_dim = model_dim as usize;
    let tap_count = if kernel_size > 0 {
        kernel_size - 1
    } else {
        0
    };

    unsafe {
        for token_idx in 0..suffix_len + kernel_size.saturating_sub(1) {
            for channel_idx in 0..model_dim {
                let tap_count = kernel_size.saturating_sub(1);
                let w_row = w.add(channel_idx * kernel_size);

                // Threads [0..suffix_len-1]: Compute outputs
                if token_idx < suffix_len {
                    let mut acc = 0.0f32;
                    if has_bias {
                        acc = (*b.unwrap().add(channel_idx)).to_f32().unwrap();
                    }

                    // Convolve using padded buffer
                    for tap in 0..kernel_size {
                        let padded_row = token_idx + tap;
                        let padded_offset = padded_row * model_dim + channel_idx;
                        let sample = (*padded.add(padded_offset)).to_f32().unwrap();
                        acc += (*w_row.add(tap)).to_f32().unwrap() * sample;
                    }

                    // Apply post-gate from in_proj
                    let in_proj_idx = token_idx * in_proj_stride + channel_idx;
                    let post_conv_gate = (*in_proj.add(in_proj_idx + model_dim)).to_f32().unwrap();
                    let gated_output = acc * post_conv_gate;

                    // Write output
                    let out_idx = token_idx * model_dim + channel_idx;
                    *out.add(out_idx) = T::from(gated_output).unwrap();
                } else if tap_count > 0 {
                    let tap = token_idx - suffix_len;
                    if tap >= tap_count {
                        return;
                    }

                    // Copy last tap_count values from padded to state_out
                    let padded_row = suffix_len + tap;
                    let padded_offset = padded_row * model_dim + channel_idx;
                    let state_idx = channel_idx * state_stride + tap;

                    *state_out.add(state_idx) = T::from(*padded.add(padded_offset)).unwrap();
                }
            }
        }
    }
}

#[kernel(ShortConvDecode)]
#[variants(T, f32, f16, bf16)]
pub fn short_conv_decode<T: ArrayElement + Float>(
    in_proj: *const T,
    w: *const T,
    #[optional(has_bias)] b: Option<*const T>,
    #[optional(!state_in_place)] state: Option<*const T>,
    out: *mut T,
    next_state: *mut T,
    suffix_len: u32,
    kernel_size: u32,
    in_proj_stride: u32,
    state_stride: u32,
    model_dim: u32,
    #[specialize] has_bias: bool,
    #[specialize] state_in_place: bool,
) {
    let suffix_len = suffix_len as usize;
    let kernel_size = kernel_size as usize;
    let in_proj_stride = in_proj_stride as usize;
    let state_stride = state_stride as usize;
    let model_dim = model_dim as usize;

    let mut state: *const T = match state_in_place {
        true => next_state,
        false => state.unwrap(),
    };

    unsafe {
        for token_idx in 0..suffix_len {
            for channel_idx in 0..model_dim {
                let tap_count = kernel_size.saturating_sub(1);
                let state_offset = channel_idx * state_stride;
                let w_row = w.add(channel_idx * kernel_size);
                let in_proj_idx = token_idx * in_proj_stride + channel_idx;
                let pre_conv_gate = (*in_proj.add(in_proj_idx)).to_f32().unwrap();
                let post_conv_gate = (*in_proj.add(in_proj_idx + model_dim)).to_f32().unwrap();
                let x_in = (*in_proj.add(in_proj_idx + 2 * model_dim)).to_f32().unwrap();
                let x = (x_in * pre_conv_gate).to_f32().unwrap();

                let mut acc = 0.0f32;
                if has_bias {
                    acc = (*b.unwrap().add(channel_idx)).to_f32().unwrap();
                }
                for tap in 0..tap_count {
                    let sample = (*state.add(state_offset + tap)).to_f32().unwrap();
                    acc += (*w_row.add(tap)).to_f32().unwrap() * sample;
                }
                acc += (*w_row.add(tap_count)).to_f32().unwrap() * x;

                let gated_output = acc * post_conv_gate;
                let out_idx = token_idx * model_dim + channel_idx;
                *out.add(out_idx) = T::from(gated_output).unwrap();

                if tap_count > 0 {
                    for tap in 0..tap_count - 1 {
                        *next_state.add(state_offset + tap) = *state.add(state_offset + tap + 1);
                    }
                    *next_state.add(state_offset + tap_count - 1) = T::from(x).unwrap();
                }
            }
            // After processing all channels for this token, subsequent tokens
            // must read from the updated state in next_state.
            if !state_in_place {
                state = next_state;
            }
        }
    }
}

#[kernel(ShortConvTrie)]
#[variants(T, f32, f16, bf16)]
pub fn short_conv_trie<T: ArrayElement + Float>(
    in_proj: *const T,
    w: *const T,
    #[optional(has_bias)] b: Option<*const T>,
    base_state: *const T,
    parents: *const i32,
    out: *mut T,
    suffix_state: *mut T,
    suffix_len: u32,
    kernel_size: u32,
    in_proj_stride: u32,
    state_stride: u32,
    model_dim: u32,
    #[specialize] has_bias: bool,
) {
    let suffix_len = suffix_len as usize;
    let kernel_size = kernel_size as usize;
    let in_proj_stride = in_proj_stride as usize;
    let state_stride = state_stride as usize;
    let model_dim = model_dim as usize;

    unsafe {
        for channel_idx in 0..model_dim {
            let tap_count = kernel_size.saturating_sub(1);
            let w_row = w.add(channel_idx * kernel_size);
            let base_state_offset = channel_idx * state_stride;

            for node in 0..suffix_len {
                let in_proj_idx = node * in_proj_stride + channel_idx;
                let pre_conv_gate = (*in_proj.add(in_proj_idx)).to_f32().unwrap();
                let post_conv_gate = (*in_proj.add(in_proj_idx + model_dim)).to_f32().unwrap();
                let x_in = (*in_proj.add(in_proj_idx + 2 * model_dim)).to_f32().unwrap();
                let x = (x_in * pre_conv_gate).to_f32().unwrap();

                // Select parent state (root uses base_state)
                let parent = *parents.add(node);
                let parent_state = if parent < 0 {
                    base_state.add(base_state_offset)
                } else {
                    let parent = parent as usize;
                    suffix_state.add((parent * model_dim + channel_idx) * state_stride)
                };

                let mut acc = 0.0f32;
                if has_bias {
                    acc = (*b.unwrap().add(channel_idx)).to_f32().unwrap();
                }
                for tap in 0..tap_count {
                    let sample = (*parent_state.add(tap)).to_f32().unwrap();
                    acc += (*w_row.add(tap)).to_f32().unwrap() * sample;
                }
                acc += (*w_row.add(tap_count)).to_f32().unwrap() * x;

                let gated_output = acc * post_conv_gate;
                *out.add(node * model_dim + channel_idx) = T::from(gated_output).unwrap();

                if tap_count > 0 {
                    let dst_state = suffix_state.add((node * model_dim + channel_idx) * state_stride);
                    for tap in 0..tap_count - 1 {
                        *dst_state.add(tap) = *parent_state.add(tap + 1);
                    }
                    *dst_state.add(tap_count - 1) = T::from(x).unwrap();
                }
            }
        }
    }
}
