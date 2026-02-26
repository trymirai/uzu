use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeExpertsDecodeSinglePassA)]
#[variants(T, f32, f16, bf16)]
pub fn moe_experts_decode_single_pass_a<T: ArrayElement + Float>(
    #[allow(unused)] x: *const T,
    #[allow(unused)] topk_ids: *const i32,
    #[allow(unused)] w13_all: *const T,
    #[allow(unused)] biases: *const T,
    #[allow(unused)] hidden_out: *mut f32,
    #[allow(unused)] d_model: u32,
    #[allow(unused)] d_ff: u32,
    #[allow(unused)] k: u32,
    #[allow(unused)] silu_alpha: f32,
    #[allow(unused)] gate_clip_min: f32,
    #[allow(unused)] gate_clip_max: f32,
    #[allow(unused)] up_clip_min: f32,
    #[allow(unused)] up_clip_max: f32,
    #[allow(unused)]
    #[specialize]
    gating_sel: u32,
) {
    todo!()
}

#[kernel(MoeExpertsDecodeSinglePassB)]
#[variants(T, f32, f16, bf16)]
pub fn moe_experts_decode_single_pass_b<T: ArrayElement + Float>(
    #[allow(unused)] hidden: *const f32,
    #[allow(unused)] topk_ids: *const i32,
    #[allow(unused)] topk_probs: *const T,
    #[allow(unused)] w2_all: *const T,
    #[allow(unused)] biases: *const T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] d_model: u32,
    #[allow(unused)] d_ff: u32,
    #[allow(unused)] k_input: u32,
) {
    todo!()
}
