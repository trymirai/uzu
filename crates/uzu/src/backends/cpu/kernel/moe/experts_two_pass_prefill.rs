use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeExpertsPrefillPassA)]
#[variants(T, f32, f16, bf16)]
pub fn moe_experts_prefill_pass_a<T: ArrayElement + Float>(
    x_perm: *const T,
    expert_offsets: *const u32,
    w13_all: *const T,
    up_biases: *const T,
    hidden_out: *mut f32,
    d_model: u32,
    d_ff: u32,
    e: u32,
    gate_clip_min: f32,
    gate_clip_max: f32,
    up_clip_min: f32,
    up_clip_max: f32,
    silu_alpha: f32,
    tile_map: *const u32,
    #[specialize] gating_sel: u32,
    __dsl_indirect_dispatch_buffer: *const u32,
) {
    let _ = (
        x_perm,
        expert_offsets,
        w13_all,
        up_biases,
        hidden_out,
        d_model,
        d_ff,
        e,
        gate_clip_min,
        gate_clip_max,
        up_clip_min,
        up_clip_max,
        silu_alpha,
        tile_map,
        gating_sel,
        __dsl_indirect_dispatch_buffer,
    );
    todo!()
}

#[kernel(MoeExpertsPrefillPassB)]
#[variants(T, f32, f16, bf16)]
pub fn moe_experts_prefill_pass_b<T: ArrayElement + Float>(
    hidden: *const f32,
    expert_offsets: *const u32,
    w2_all: *const T,
    down_biases: *const T,
    output: *mut T,
    d_model: u32,
    d_ff: u32,
    e: u32,
    tile_map: *const u32,
    __dsl_indirect_dispatch_buffer: *const u32,
) {
    let _ = (
        hidden,
        expert_offsets,
        w2_all,
        down_biases,
        output,
        d_model,
        d_ff,
        e,
        tile_map,
        __dsl_indirect_dispatch_buffer,
    );
    todo!()
}
