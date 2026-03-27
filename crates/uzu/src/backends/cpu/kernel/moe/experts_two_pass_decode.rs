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
