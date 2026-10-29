use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use super::common::compute_route_gemv;
use crate::{
    backends::{
        common::gpu_types::{
            gemm::{GemmBPrologueKind, GemmDTransform},
            hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        },
        cpu::kernel::activation_transform::hadamard_transform,
    },
    data_type::DataTypeRepr,
};

#[kernel(MoeDownCombine)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 0, 32, 64)]
#[variants(BITS, 0, 4, 8)]
#[constraint((BITS == 0) == (GROUP_SIZE == 0))]
pub fn moe_down_combine<T: DataTypeRepr + Float, const GROUP_SIZE: u32, const BITS: u32>(
    b: *const u8,
    #[optional(b_prologue != GemmBPrologueKind::FullPrecision)] scales: Option<*const T>,
    #[optional(b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)] zero_points: Option<*const u8>,
    #[optional(b_prologue == GemmBPrologueKind::ScaleBiasDequant)] biases: Option<*const T>,
    #[optional(has_shared)] shared_b: Option<*const u8>,
    #[optional(has_shared && b_prologue != GemmBPrologueKind::FullPrecision)] shared_scales: Option<*const T>,
    #[optional(has_shared && b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)] shared_zero_points: Option<
        *const u8,
    >,
    #[optional(has_shared && b_prologue == GemmBPrologueKind::ScaleBiasDequant)] shared_biases: Option<*const T>,
    a: *const T,
    expert_ids: *const u32,
    expert_weights: *const f32,
    #[optional(has_gate)] shared_gate: Option<*const T>,
    output: *mut T,
    #[optional(output_transform.contains(GemmDTransform::BIAS))] output_bias: Option<*const T>,
    #[optional(output_transform.contains(GemmDTransform::RHT))] hadamard_factors: Option<*const i32>,
    #[optional(has_shared && output_transform.contains(GemmDTransform::BIAS))] shared_output_bias: Option<*const T>,
    #[allow(unused)]
    #[optional(has_shared && output_transform.contains(GemmDTransform::RHT))]
    shared_hadamard_factors: Option<*const i32>,
    in_vec_size: u32,
    out_vec_size: u32,
    active_experts_per_token: u32,
    #[allow(unused)] group_count_x: u32,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
    zero_point_output_stride: u32,
    zero_point_group_stride: u32,
    batch_size: u32,
    gate_stride: u32,
    #[specialize] b_prologue: GemmBPrologueKind,
    #[specialize] output_transform: GemmDTransform,
    #[allow(unused)]
    #[specialize]
    input_aligned: bool,
    #[specialize] signed_codes: bool,
    #[allow(unused)]
    #[specialize]
    full_tile: bool,
    #[specialize] has_shared: bool,
    #[specialize] has_gate: bool,
) {
    let k_u = in_vec_size as usize;
    let n_u = out_vec_size as usize;
    let active_experts = active_experts_per_token as usize;
    let gate_stride_u = gate_stride as usize;
    let num_routed_routes = (batch_size * active_experts_per_token) as usize;

    for token_idx in 0..batch_size as usize {
        let mut combined = vec![0.0f32; n_u];
        let base_routed_idx = token_idx * active_experts;

        for k in 0..active_experts {
            let route_idx = base_routed_idx + k;
            let expert_id = unsafe { *expert_ids.add(route_idx) } as usize;
            let weight = unsafe { *expert_weights.add(route_idx) };

            let expert_b = unsafe { b.add(expert_id * b_expert_stride_bytes as usize) };
            let expert_scales = scales.map(|s| unsafe { s.add(expert_id * scale_expert_stride as usize) });
            let expert_zp = zero_points.map(|z| unsafe { z.add(expert_id * zero_point_expert_stride as usize) });
            let expert_biases = biases.map(|b| unsafe { b.add(expert_id * scale_expert_stride as usize) });
            let route_a = unsafe { a.add(route_idx * k_u) };

            let mut row_out = compute_route_gemv::<T, GROUP_SIZE, BITS>(
                expert_b,
                expert_scales,
                expert_zp,
                expert_biases,
                route_a,
                k_u,
                n_u,
                scale_output_stride as usize,
                scale_group_stride as usize,
                zero_point_output_stride as usize,
                zero_point_group_stride as usize,
                signed_codes,
                b_prologue,
            );

            if output_transform.contains(GemmDTransform::BIAS) {
                let bias_ptr = output_bias.expect("BIAS requires output_bias");
                let expert_bias = unsafe { bias_ptr.add(expert_id * n_u) };
                for (col, val) in row_out.iter_mut().enumerate() {
                    let b_val = unsafe { *expert_bias.add(col) }.to_f32().unwrap();
                    *val += b_val;
                }
            }

            for (col, &val) in row_out.iter().enumerate() {
                combined[col] += weight * val;
            }
        }

        if has_shared {
            let shared_route_idx = num_routed_routes + token_idx;
            let expert_b = shared_b.unwrap();
            let expert_scales = shared_scales;
            let expert_zp = shared_zero_points;
            let expert_biases = shared_biases;
            let route_a = unsafe { a.add(shared_route_idx * k_u) };

            let mut row_out = compute_route_gemv::<T, GROUP_SIZE, BITS>(
                expert_b,
                expert_scales,
                expert_zp,
                expert_biases,
                route_a,
                k_u,
                n_u,
                scale_output_stride as usize,
                scale_group_stride as usize,
                zero_point_output_stride as usize,
                zero_point_group_stride as usize,
                signed_codes,
                b_prologue,
            );

            if output_transform.contains(GemmDTransform::BIAS) {
                let bias_ptr = shared_output_bias.expect("BIAS requires shared_output_bias");
                for (col, val) in row_out.iter_mut().enumerate() {
                    let b_val = unsafe { *bias_ptr.add(col) }.to_f32().unwrap();
                    *val += b_val;
                }
            }

            let shared_w = if has_gate {
                let g = unsafe { *shared_gate.unwrap().add(token_idx * gate_stride_u) }.to_f32().unwrap();
                1.0f32 / (1.0f32 + (-g).exp())
            } else {
                1.0f32
            };

            for (col, &val) in row_out.iter().enumerate() {
                combined[col] += shared_w * val;
            }
        }

        if output_transform.contains(GemmDTransform::RHT) {
            let factors = hadamard_factors.expect("RHT requires hadamard_factors");
            let block_size = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
            for block_start in (0..n_u).step_by(block_size) {
                let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize];
                stripe[..block_size].copy_from_slice(&combined[block_start..(block_start + block_size)]);
                hadamard_transform(&mut stripe);
                for lane in 0..block_size {
                    let sign = unsafe { *factors.add(block_start + lane) } as f32;
                    combined[block_start + lane] = stripe[lane] * sign;
                }
            }
        }

        for (col, &val) in combined.iter().enumerate() {
            unsafe {
                *output.add(token_idx * n_u + col) = T::from(val).unwrap();
            }
        }
    }
}
