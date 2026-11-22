use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::moe::MoeExpertFormat,
        cpu::kernel::moe::expert_math::{ExpertMatrix, dot, expert_activation},
    },
    data_type::DataTypeRepr,
};

#[kernel(MoeGroupedUp)]
#[variants(T, bf16)]
pub fn moe_grouped_up<T: DataTypeRepr + Float>(
    input: *const T,
    weights: *const u8,
    scales: *const u8,
    #[optional(format == MoeExpertFormat::Nvfp4)] global_scales: Option<*const f32>,
    expert_counts: *const u32,
    expert_routes: *const u32,
    route_weights: *const f32,
    hidden: *mut f16,
    tokens: u32,
    experts: u32,
    model_dim: u32,
    hidden_dim: u32,
    top_k: u32,
    up_soft_cap: f32,
    gate_soft_cap: f32,
    activation_alpha: f32,
    #[specialize] format: MoeExpertFormat,
    #[specialize] cap_up: bool,
    #[specialize] cap_gate: bool,
) {
    assert_eq!(global_scales.is_some(), format == MoeExpertFormat::Nvfp4);
    let (tokens, model_dim, hidden_dim, top_k) =
        (tokens as usize, model_dim as usize, hidden_dim as usize, top_k as usize);
    for expert in 0..experts as usize {
        let count = unsafe { *expert_counts.add(expert) } as usize;
        if count == 0 {
            continue;
        }
        let matrix =
            unsafe { ExpertMatrix::new(format, weights, scales, global_scales, expert, 2 * hidden_dim, model_dim) };
        let routes: Vec<usize> =
            (0..count).map(|index| unsafe { *expert_routes.add(expert * tokens + index) } as usize).collect();
        let inputs: Vec<Vec<f32>> = routes
            .iter()
            .map(|&route| {
                (0..model_dim)
                    .map(|col| unsafe { *input.add(route / top_k * model_dim + col) }.to_f32().unwrap())
                    .collect()
            })
            .collect();
        for unit in 0..hidden_dim {
            let up_row = matrix.row(unit);
            let gate_row = matrix.row(hidden_dim + unit);
            for (&route, values) in routes.iter().zip(&inputs) {
                let activated = expert_activation(
                    dot(&up_row, values),
                    dot(&gate_row, values),
                    cap_up.then_some(up_soft_cap),
                    cap_gate.then_some(gate_soft_cap),
                    activation_alpha,
                );
                let route_weight = unsafe { *route_weights.add(route) };
                unsafe {
                    *hidden.add(route * hidden_dim + unit) = f16::from_f32(activated * route_weight);
                }
            }
        }
    }
}
