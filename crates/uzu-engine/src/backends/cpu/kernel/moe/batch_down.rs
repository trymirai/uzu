use half::f16;
use uzu_engine_macros::kernel;

use crate::backends::{
    common::gpu_types::moe::MoeExpertFormat,
    cpu::kernel::moe::expert_math::{ExpertMatrix, active_expert_routes, dot},
};

#[kernel(MoeExpertDownBatch)]
#[variants(MAX_PASS, 2, 4)]
pub fn moe_expert_down_batch<const MAX_PASS: u32>(
    hidden: *const f16,
    weights: *const u8,
    params: *const u32,
    expert_routes: *const u32,
    active_experts: *const u32,
    route_outputs: *mut f32,
    tokens: u32,
    model_dim: u32,
    hidden_dim: u32,
    experts: u32,
    top_k: u32,
) {
    let _ = (experts, top_k);
    let (tokens, model_dim, hidden_dim) = (tokens as usize, model_dim as usize, hidden_dim as usize);
    for (expert, routes) in unsafe { active_expert_routes(active_experts, expert_routes, tokens) } {
        let matrix = unsafe {
            ExpertMatrix::new(MoeExpertFormat::AffineU4, weights, params.cast(), None, expert, model_dim, hidden_dim)
        };
        let inputs: Vec<Vec<f32>> = routes
            .iter()
            .map(|&route| {
                (0..hidden_dim).map(|col| unsafe { *hidden.add(route * hidden_dim + col) }.to_f32()).collect()
            })
            .collect();
        for row in 0..model_dim {
            let weights_row = matrix.row(row);
            for (&route, values) in routes.iter().zip(&inputs) {
                unsafe {
                    *route_outputs.add(route * model_dim + row) = dot(&weights_row, values);
                }
            }
        }
    }
}
