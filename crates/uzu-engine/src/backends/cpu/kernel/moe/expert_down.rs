use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::moe::MoeExpertFormat,
        cpu::kernel::moe::expert_math::{ExpertMatrix, active_expert_routes, dot},
    },
    data_type::DataTypeRepr,
};

#[kernel(MoeExpertDown)]
#[variants(T, bf16)]
#[variants(MAX_TOKENS, 1, 8, 32)]
pub fn moe_expert_down<T: DataTypeRepr + Float, const MAX_TOKENS: u32>(
    hidden: *const f16,
    weights: *const u8,
    scales: *const u8,
    #[optional(format == MoeExpertFormat::Nvfp4)] global_scales: Option<*const f32>,
    expert_routes: *const u32,
    active_experts: *const u32,
    output: *mut T,
    tokens: u32,
    model_dim: u32,
    hidden_dim: u32,
    top_k: u32,
    #[specialize] format: MoeExpertFormat,
) {
    assert_eq!(global_scales.is_some(), format == MoeExpertFormat::Nvfp4);
    assert!(tokens <= MAX_TOKENS);
    let (tokens, model_dim, hidden_dim, top_k) =
        (tokens as usize, model_dim as usize, hidden_dim as usize, top_k as usize);
    let mut sums = vec![0.0f32; tokens * model_dim];
    for (expert, routes) in unsafe { active_expert_routes(active_experts, expert_routes, tokens) } {
        let matrix =
            unsafe { ExpertMatrix::new(format, weights, scales, global_scales, expert, model_dim, hidden_dim) };
        let inputs: Vec<Vec<f32>> = routes
            .iter()
            .map(|&route| {
                (0..hidden_dim).map(|col| unsafe { *hidden.add(route * hidden_dim + col) }.to_f32()).collect()
            })
            .collect();
        for row in 0..model_dim {
            let weights_row = matrix.row(row);
            for (&route, values) in routes.iter().zip(&inputs) {
                sums[route / top_k * model_dim + row] += dot(&weights_row, values);
            }
        }
    }
    for (index, &sum) in sums.iter().enumerate() {
        unsafe {
            *output.add(index) = T::from(sum).unwrap();
        }
    }
}
