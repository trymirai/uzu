use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::ActivationType};

#[kernel(ConvTreeScan)]
#[variants(T, f32, bf16)]
pub fn conv_tree_scan<T: ArrayElement + Float>(
    in_proj: *const T,
    conv_weight: *const f32,
    #[optional(has_bias)] bias: Option<*const f32>,
    base_state: *const f32,
    parents: *const i32,
    out_proj: *mut T,
    suffix_state: *mut f32,
    suffix_len: u32,
    #[specialize] kernel_size: u32,
    total_proj_dim: u32,
    conv_dim: u32,
    #[specialize] has_bias: bool,
) {
    let suffix_len = suffix_len as usize;
    let kernel_size = kernel_size as usize;
    let total_proj_dim = total_proj_dim as usize;
    let conv_dim = conv_dim as usize;
    let state_stride = kernel_size - 1;

    for node_idx in 0..suffix_len {
        for channel_idx in 0..total_proj_dim {
            let proj_idx = node_idx * total_proj_dim + channel_idx;
            if channel_idx >= conv_dim {
                unsafe { *out_proj.add(proj_idx) = *in_proj.add(proj_idx) };
                continue;
            }

            let mut acc = if has_bias {
                unsafe { *bias.unwrap().add(channel_idx) }
            } else {
                0.0
            };
            let weight_offset = channel_idx * kernel_size;
            let base_state_offset = channel_idx * state_stride;
            let mut source_row = node_idx as i32;
            for history_offset in 0..kernel_size {
                let sample = if source_row >= 0 {
                    let source_proj_idx = source_row as usize * total_proj_dim + channel_idx;
                    unsafe { (*in_proj.add(source_proj_idx)).to_f32().unwrap() }
                } else {
                    let base_state_tap = state_stride - source_row.unsigned_abs() as usize;
                    unsafe { *base_state.add(base_state_offset + base_state_tap) }
                };
                let weight_tap = kernel_size - 1 - history_offset;
                acc += sample * unsafe { *conv_weight.add(weight_offset + weight_tap) };

                if history_offset < state_stride {
                    let state_tap = state_stride - 1 - history_offset;
                    let state_idx = (node_idx * conv_dim + channel_idx) * state_stride + state_tap;
                    unsafe { *suffix_state.add(state_idx) = sample };
                }

                source_row = if source_row >= 0 {
                    unsafe { *parents.add(source_row as usize) }
                } else {
                    source_row - 1
                };
            }

            unsafe { *out_proj.add(proj_idx) = T::from(ActivationType::SILU.activate(acc)).unwrap() };
        }
    }
}
