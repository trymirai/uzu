use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeExpertsDecodePassA)]
#[variants(T, f32, f16, bf16)]
pub fn moe_experts_decode_pass_a<T: ArrayElement + Float>(
    #[allow(unused)] x_perm: *const T,
    #[allow(unused)] expert_offsets: *const u32,
    #[allow(unused)] w13_all: *const T,
    #[allow(unused)] hidden_out: *mut f32,
    #[allow(unused)] up_biases: *const T,
    #[allow(unused)] d_model: u32,
    #[allow(unused)] d_ff: u32,
    #[allow(unused)] e: u32,
    #[allow(unused)] gate_clip_min: f32,
    #[allow(unused)] gate_clip_max: f32,
    #[allow(unused)] up_clip_min: f32,
    #[allow(unused)] up_clip_max: f32,
    #[allow(unused)] silu_alpha: f32,
    #[allow(unused)] tile_map: *const u32,
    #[allow(unused)]
    #[specialize]
    gating_sel: u32,
    #[allow(unused)] __dsl_indirect_dispatch_buffer: *const u32,
) {
    todo!()
}

#[kernel(MoeExpertsDecodeDownFused2D)]
#[variants(T, f32, f16, bf16)]
#[variants(AccumT, f32)]
pub fn moe_experts_decode_down_fused2_d<T: ArrayElement + Float, AccumT: ArrayElement + Float>(
    #[allow(unused)] hidden: *const f32,
    #[allow(unused)] row_expert_map: *const u32,
    #[allow(unused)] w2_all: *const T,
    #[allow(unused)] down_biases: *const T,
    #[allow(unused)] y_out: *mut T,
    #[allow(unused)] total_rows: u32,
    #[allow(unused)] d_model: u32,
    #[allow(unused)] d_ff: u32,
    #[allow(unused)] e: u32,
) {
    todo!()
}
