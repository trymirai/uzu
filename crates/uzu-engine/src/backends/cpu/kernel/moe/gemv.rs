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

#[kernel(MoeGemv)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 0, 32, 64)]
#[variants(BITS, 0, 4, 8)]
#[constraint((BITS == 0) == (GROUP_SIZE == 0))]
pub fn moe_gemv<T: DataTypeRepr + Float, const GROUP_SIZE: u32, const BITS: u32>(
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
    d: *mut T,
    #[optional(output_transform.contains(GemmDTransform::BIAS))] output_bias: Option<*const T>,
    #[optional(output_transform.contains(GemmDTransform::RHT))] hadamard_factors: Option<*const i32>,
    #[optional(has_shared && output_transform.contains(GemmDTransform::BIAS))] shared_output_bias: Option<*const T>,
    #[optional(has_shared && output_transform.contains(GemmDTransform::RHT))] shared_hadamard_factors: Option<
        *const i32,
    >,
    in_vec_size: u32,
    out_vec_size: u32,
    num_routes: u32,
    active_experts_per_token: u32,
    #[allow(unused)] group_count_x: u32,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
    zero_point_output_stride: u32,
    zero_point_group_stride: u32,
    num_routed_routes: u32,
    #[specialize] b_prologue: GemmBPrologueKind,
    #[specialize] output_transform: GemmDTransform,
    #[allow(unused)]
    #[specialize]
    input_aligned: bool,
    #[specialize] signed_codes: bool,
    #[allow(unused)]
    #[specialize]
    full_tile: bool,
    #[specialize] a_per_token: bool,
    #[specialize] has_shared: bool,
) {
    let k_u = in_vec_size as usize;
    let n_u = out_vec_size as usize;
    let routes = num_routes as usize;
    let top_k = active_experts_per_token as usize;
    let _bits = BITS as usize;
    let _group_size = GROUP_SIZE as usize;

    for route_idx in 0..routes {
        let is_shared = has_shared && (route_idx >= num_routed_routes as usize);
        let expert_id = if is_shared {
            0
        } else {
            unsafe { *expert_ids.add(route_idx) as usize }
        };
        let a_row_idx = if a_per_token {
            if is_shared {
                route_idx - num_routed_routes as usize
            } else {
                route_idx / top_k
            }
        } else {
            route_idx
        };

        let expert_b = if is_shared {
            shared_b.unwrap()
        } else {
            unsafe { b.add(expert_id * b_expert_stride_bytes as usize) }
        };
        let expert_scales = if is_shared {
            shared_scales
        } else {
            scales.map(|ptr| unsafe { ptr.add(expert_id * scale_expert_stride as usize) })
        };
        let expert_zero_points = if is_shared {
            shared_zero_points
        } else {
            zero_points.map(|ptr| unsafe { ptr.add(expert_id * zero_point_expert_stride as usize) })
        };
        let expert_biases = if is_shared {
            shared_biases
        } else {
            biases.map(|ptr| unsafe { ptr.add(expert_id * scale_expert_stride as usize) })
        };
        let route_a = unsafe { a.add(a_row_idx * k_u) };
        let route_d = unsafe { d.add(route_idx * n_u) };
        let expert_output_bias = if is_shared {
            shared_output_bias
        } else {
            output_bias.map(|ptr| unsafe { ptr.add(expert_id * n_u) })
        };
        let expert_hadamard = if is_shared {
            shared_hadamard_factors
        } else {
            hadamard_factors.map(|ptr| unsafe { ptr.add(expert_id * n_u) })
        };

        let mut row_out = compute_route_gemv::<T, GROUP_SIZE, BITS>(
            expert_b,
            expert_scales,
            expert_zero_points,
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

        if output_transform.contains(GemmDTransform::RHT) {
            let factors = expert_hadamard.expect("RHT requires hadamard_factors");
            let block_size = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
            for block_start in (0..n_u).step_by(block_size) {
                let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize];
                for lane in 0..block_size {
                    stripe[lane] = T::from(row_out[block_start + lane]).unwrap().to_f32().unwrap();
                }
                hadamard_transform(&mut stripe);
                for lane in 0..block_size {
                    let sign = unsafe { *factors.add(block_start + lane) } as f32;
                    let transformed = T::from(stripe[lane] * sign).unwrap().to_f32().unwrap();
                    row_out[block_start + lane] = transformed;
                }
            }
        }

        if output_transform.contains(GemmDTransform::BIAS) {
            let bias_ptr = expert_output_bias.expect("BIAS requires output_bias");
            for (col, val) in row_out.iter_mut().enumerate() {
                let b_val = unsafe { *bias_ptr.add(col) }.to_f32().unwrap();
                *val += b_val;
            }
        }

        for (col, &val) in row_out.iter().enumerate() {
            unsafe {
                *route_d.add(col) = T::from(val).unwrap();
            }
        }
    }
}
