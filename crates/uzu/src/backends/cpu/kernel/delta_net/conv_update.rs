use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;
use crate::backends::cpu::kernel::activation::silu_f32;

// Single-token causal conv1d with SiLU, in-place.
#[kernel(DeltaNetConvUpdate)]
#[variants(T, f32, f16, bf16)]
pub fn delta_net_conv_update<T: ArrayElement + Float>(
    #[allow(unused)] w: *const T,
    #[allow(unused)]
    #[optional(has_bias)]
    bias: Option<*const T>,
    #[allow(unused)] in_out: *mut T,
    #[allow(unused)] state: *mut T,
    #[allow(unused)] kernel_size: u32,
    #[allow(unused)] conv_dim: u32,
    #[allow(unused)] state_stride: u32,
    #[allow(unused)]
    #[specialize]
    has_bias: bool,
) {
    let tap_count = kernel_size as usize - 1;

    for c in 0..conv_dim as usize {
        let state_offset = c * state_stride as usize;
        let w_offset = c * kernel_size as usize;

        let x = unsafe { (*in_out.add(c)).to_f32().unwrap() };

        let mut acc = if has_bias {
            unsafe { (*bias.unwrap().add(c)).to_f32().unwrap() }
        } else {
            0.0f32
        };

        for tap in 0..tap_count {
            acc += unsafe { (*state.add(state_offset + tap)).to_f32().unwrap() }
                * unsafe { (*w.add(w_offset + tap)).to_f32().unwrap() };
        }
        acc += x * unsafe { (*w.add(w_offset + tap_count)).to_f32().unwrap() };

        unsafe {
            *in_out.add(c) = T::from(silu_f32(acc)).unwrap();
        }

        for tap in 1..tap_count {
            unsafe {
                *state.add(state_offset + tap - 1) = *state.add(state_offset + tap);
            }
        }
        unsafe {
            *state.add(state_offset + tap_count - 1) = T::from(x).unwrap();
        }
    }
}
