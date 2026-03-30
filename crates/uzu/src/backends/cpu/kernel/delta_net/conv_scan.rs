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
    let sl = suffix_len as usize;
    let ks = kernel_size as usize;
    let rs = row_stride as usize;
    let ss = state_stride as usize;
    let cd = conv_dim as usize;
    let os = out_stride as usize;

    for token in 0..sl {
        for ch in 0..cd {
            let w_offset = ch * ks;

            let mut acc = if has_bias {
                unsafe { (*bias.unwrap().add(ch)).to_f32().unwrap() }
            } else {
                0.0f32
            };

            for tap in 0..ks {
                let padded_row = token + tap;
                let padded_idx = padded_row * rs + ch;
                let sample = unsafe { (*conv_padded.add(padded_idx)).to_f32().unwrap() };
                acc += unsafe { (*conv_weight.add(w_offset + tap)).to_f32().unwrap() } * sample;
            }

            unsafe {
                *in_proj.add(token * os + ch) = T::from(ActivationType::SILU.activate(acc)).unwrap();
            }
        }
    }

    // state_out layout: [conv_dim, state_stride] (channel-major)
    for ch in 0..cd {
        for tap in 0..ss {
            let padded_row = sl + tap;
            let padded_idx = padded_row * rs + ch;
            let sample = unsafe { (*conv_padded.add(padded_idx)).to_f32().unwrap() };
            unsafe {
                *state_out.add(ch * ss + tap) = T::from(sample).unwrap();
            }
        }
    }
}
