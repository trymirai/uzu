use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AttentionTwoPass1)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_two_pass1<T: ArrayElement + Float, const HEAD_DIM: u32>(
    #[allow(unused)] queries: *const T,
    #[allow(unused)] keys: *const T,
    #[allow(unused)] values: *const T,
    #[allow(unused)] out: *mut f32,
    #[allow(unused)] sums: *mut f32,
    #[allow(unused)] maxs: *mut f32,
    #[allow(unused)] gqa_factor: u32,
    #[allow(unused)] sequence_length: u32,
    #[allow(unused)] k_head_stride: u32,
    #[allow(unused)] k_seq_stride: u32,
    #[allow(unused)] v_head_stride: u32,
    #[allow(unused)] v_seq_stride: u32,
    #[allow(unused)] scale: f32,
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] suffix_length: u32,
    #[allow(unused)]
    #[optional(float_mask)]
    fmask: Option<*const T>,
    #[allow(unused)]
    #[optional(has_mask)]
    mask_kv_seq_stride: Option<u32>,
    #[allow(unused)]
    #[optional(has_mask)]
    mask_q_seq_stride: Option<u32>,
    #[allow(unused)]
    #[optional(has_mask)]
    mask_head_stride: Option<u32>,
    #[allow(unused)]
    #[optional(has_sinks)]
    sinks: Option<*const f32>,
    #[allow(unused)]
    #[specialize]
    float_mask: bool,
    #[allow(unused)]
    #[specialize]
    has_mask: bool,
    #[allow(unused)]
    #[specialize]
    has_sinks: bool,
    #[allow(unused)]
    #[specialize]
    do_causal: bool,
) {
    todo!()
}

#[kernel(AttentionTwoPass2)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_two_pass2<T: ArrayElement + Float, const HEAD_DIM: u32>(
    #[allow(unused)] partials: *const f32,
    #[allow(unused)] sums: *const f32,
    #[allow(unused)] maxs: *const f32,
    #[allow(unused)] out: *mut T,
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] suffix_length: u32,
) {
    todo!()
}
