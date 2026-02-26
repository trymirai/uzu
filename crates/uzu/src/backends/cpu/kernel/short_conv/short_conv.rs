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
    todo!()
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
