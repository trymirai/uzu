use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;
use crate::backends::cpu::kernel::activation::silu_f32;

// Multi-token causal conv1d with SiLU for DeltaNet.
// Reads from padded buffer (Conv1dPack output), writes conv'd+SiLU'd values
// back to in_out with stride out_stride, updates conv state.
//
// padded:    [state_stride + suffix_len, row_stride] — packed history + input
// w:         [conv_dim, kernel_size]
// bias:      [conv_dim] (optional)
// in_out:    [suffix_len, out_stride] — first conv_dim channels overwritten
// state_out: [conv_dim, state_stride]
#[kernel(DeltaNetConvScan)]
#[variants(T, f32, f16, bf16)]
pub fn delta_net_conv_scan<T: ArrayElement + Float>(
    #[allow(unused)] padded: *const T,
    #[allow(unused)] w: *const T,
    #[allow(unused)]
    #[optional(has_bias)]
    bias: Option<*const T>,
    #[allow(unused)] in_out: *mut T,
    #[allow(unused)] state_out: *mut T,
    #[allow(unused)] suffix_len: u32,
    #[allow(unused)] kernel_size: u32,
    #[allow(unused)] row_stride: u32,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)] conv_dim: u32,
    #[allow(unused)] out_stride: u32,
    #[allow(unused)]
    #[specialize]
    has_bias: bool,
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
                let sample = unsafe { (*padded.add(padded_idx)).to_f32().unwrap() };
                acc += unsafe { (*w.add(w_offset + tap)).to_f32().unwrap() } * sample;
            }

            unsafe {
                *in_out.add(token * os + ch) = T::from(silu_f32(acc)).unwrap();
            }
        }
    }

    // state_out layout: [conv_dim, state_stride] (channel-major)
    for ch in 0..cd {
        for tap in 0..ss {
            let padded_row = sl + tap;
            let padded_idx = padded_row * rs + ch;
            let sample = unsafe { (*padded.add(padded_idx)).to_f32().unwrap() };
            unsafe {
                *state_out.add(ch * ss + tap) = T::from(sample).unwrap();
            }
        }
    }
}
