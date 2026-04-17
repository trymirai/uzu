use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::ActivationType};

// Multi-token causal conv1d with SiLU for DeltaNet.
// Reads from conv_padded (Conv1dPack output), writes conv'd+SiLU'd values
// back to in_proj with stride out_stride, updates conv state.
//
// conv_padded: [state_stride + suffix_len, row_stride] — packed history + input
// conv_weight: [conv_dim, kernel_size]
// bias:      [conv_dim] (optional)
// in_proj:    [suffix_len, out_stride] — first conv_dim channels overwritten
// state_out: [conv_dim, state_stride]
#[kernel(DeltaNetConvScan)]
#[variants(T, f32, f16, bf16)]
pub fn delta_net_conv_scan<T: ArrayElement + Float>(
    conv_padded: *const T,
    conv_weight: *const T,
    #[optional(has_bias)] bias: Option<*const T>,
    in_proj: *mut T,
    state_out: *mut T,
    suffix_len: u32,
    kernel_size: u32,
    row_stride: u32,
    state_stride: u32,
    conv_dim: u32,
    out_stride: u32,
    #[specialize] has_bias: bool,
) {
    let suffix_len = suffix_len as usize;
    let kernel_size = kernel_size as usize;
    let row_stride = row_stride as usize;
    let state_stride = state_stride as usize;
    let conv_dim = conv_dim as usize;
    let out_stride = out_stride as usize;

    for token in 0..suffix_len {
        for channel in 0..conv_dim {
            let w_offset = channel * kernel_size;

            let mut acc = if has_bias {
                unsafe { (*bias.unwrap().add(channel)).to_f32().unwrap() }
            } else {
                0.0f32
            };

            for tap in 0..kernel_size {
                let padded_row = token + tap;
                let padded_idx = padded_row * row_stride + channel;
                let sample = unsafe { (*conv_padded.add(padded_idx)).to_f32().unwrap() };
                acc += unsafe { (*conv_weight.add(w_offset + tap)).to_f32().unwrap() } * sample;
            }

            unsafe {
                *in_proj.add(token * out_stride + channel) = T::from(ActivationType::SILU.activate(acc)).unwrap();
            }
        }
    }

    // state_out layout: [conv_dim, state_stride] (channel-major)
    for channel in 0..conv_dim {
        for tap in 0..state_stride {
            let padded_row = suffix_len + tap;
            let padded_idx = padded_row * row_stride + channel;
            let sample = unsafe { (*conv_padded.add(padded_idx)).to_f32().unwrap() };
            unsafe {
                *state_out.add(channel * state_stride + tap) = T::from(sample).unwrap();
            }
        }
    }
}
