use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

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
    todo!()
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
    todo!()
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
    todo!()
}
