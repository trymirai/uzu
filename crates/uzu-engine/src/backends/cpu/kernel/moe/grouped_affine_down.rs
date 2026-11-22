use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::moe::MoeExpertFormat,
        cpu::kernel::moe::expert_math::{ExpertMatrix, dot},
    },
    data_type::DataTypeRepr,
};

#[kernel(MoeGroupedAffineDown)]
#[variants(T, bf16)]
pub fn moe_grouped_affine_down<T: DataTypeRepr + Float>(
    hidden: *const f16,
    weights: *const u8,
    params: *const u32,
    expert_counts: *const u32,
    expert_routes: *const u32,
    expert_outputs: *mut T,
    tokens: u32,
    experts: u32,
    model_dim: u32,
    hidden_dim: u32,
) {
    let (tokens, model_dim, hidden_dim) = (tokens as usize, model_dim as usize, hidden_dim as usize);
    for expert in 0..experts as usize {
        let count = unsafe { *expert_counts.add(expert) } as usize;
        if count == 0 {
            continue;
        }
        let matrix = unsafe {
            ExpertMatrix::new(MoeExpertFormat::AffineU4, weights, params.cast(), None, expert, model_dim, hidden_dim)
        };
        let routes: Vec<usize> =
            (0..count).map(|index| unsafe { *expert_routes.add(expert * tokens + index) } as usize).collect();
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
                    *expert_outputs.add(route * model_dim + row) = T::from(dot(&weights_row, values)).unwrap();
                }
            }
        }
    }
}
