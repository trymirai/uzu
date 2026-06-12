use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::ActivationType};

// Single-token causal conv1d with SiLU, in-place.
#[kernel(DeltaNetConvUpdate)]
#[variants(T, f32, f16, bf16)]
pub fn delta_net_conv_update<T: ArrayElement + Float>(
    conv_weight: *const f32,
    #[optional(has_bias)] bias: Option<*const f32>,
    in_out: *mut T,
    state: *mut f32,
    kernel_size: u32,
    conv_dim: u32,
    state_stride: u32,
    #[specialize] has_bias: bool,
) {
    let kernel_size = kernel_size as usize;
    let conv_dim = conv_dim as usize;
    let state_stride = state_stride as usize;
    let tap_count = kernel_size - 1;

    for channel in 0..conv_dim {
        let state_offset = channel * state_stride;
        let weight_offset = channel * kernel_size;

        let x = unsafe { (*in_out.add(channel)).to_f32().unwrap() };

        let mut acc = if has_bias {
            unsafe { *bias.unwrap().add(channel) }
        } else {
            0.0f32
        };

        for tap in 0..tap_count {
            acc += unsafe { *state.add(state_offset + tap) } * unsafe { *conv_weight.add(weight_offset + tap) };
        }
        acc += x * unsafe { *conv_weight.add(weight_offset + tap_count) };

        unsafe {
            *in_out.add(channel) = T::from(ActivationType::SILU.activate(acc)).unwrap();
        }

        for tap in 1..tap_count {
            unsafe {
                *state.add(state_offset + tap - 1) = *state.add(state_offset + tap);
            }
        }
        unsafe {
            *state.add(state_offset + tap_count - 1) = x;
        }
    }
}
