use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeExpertsDecodePassA)]
#[variants(T, f32, f16, bf16)]
pub fn moe_experts_decode_pass_a<T: ArrayElement + Float>(
    x_perm: *const T,
    expert_offsets: *const u32,
    w13_all: *const T,
    hidden_out: *mut f32,
    up_biases: *const T,
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
        hidden_out,
        up_biases,
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

#[kernel(MoeExpertsDecodeDownFused2D)]
#[variants(T, f32, f16, bf16)]
#[variants(AccumT, f32)]
pub fn moe_experts_decode_down_fused2_d<T: ArrayElement + Float, AccumT: ArrayElement + Float>(
    hidden: *const f32,
    row_expert_map: *const u32,
    w2_all: *const T,
    down_biases: *const T,
    y_out: *mut T,
    total_rows: u32,
    d_model: u32,
    d_ff: u32,
    e: u32,
) {
    let _ = e;
    let dm = d_model as usize;
    let df = d_ff as usize;

    for row_idx in 0..total_rows as usize {
        let expert_idx = unsafe { *row_expert_map.add(row_idx) } as usize;
        let hidden_base = row_idx * df;
        let w2_expert_base = expert_idx * dm * df;

        for my_col in 0..dm {
            let w2_col_base = w2_expert_base + my_col * df;

            let mut acc = 0.0f32;
            for h in 0..df {
                let h_val = unsafe { *hidden.add(hidden_base + h) };
                let w_val = unsafe { *w2_all.add(w2_col_base + h) }.to_f32().unwrap();
                acc = h_val.mul_add(w_val, acc);
            }

            // Add bias
            let bias_idx = expert_idx * dm + my_col;
            acc += unsafe { *down_biases.add(bias_idx) }.to_f32().unwrap();

            let out_idx = row_idx * dm + my_col;
            unsafe {
                *y_out.add(out_idx) = T::from(acc).unwrap();
            }
        }
    }
}
