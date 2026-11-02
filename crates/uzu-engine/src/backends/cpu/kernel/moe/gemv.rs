use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        cpu::kernel::{
            activation_transform::hadamard_transform,
            dequant::{dequantize_weight, read_zero_point},
        },
    },
    data_type::DataTypeRepr,
};

#[kernel(MoeGemv)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 64)]
#[variants(BITS, 4, 8)]
pub fn moe_gemv<T: DataTypeRepr + Float, const GROUP_SIZE: u32, const BITS: u32>(
    b: *const u8,
    scales: *const T,
    #[optional(BITS == 4)] zero_points: Option<*const u8>,
    a: *const T,
    expert_ids: *const u32,
    d: *mut T,
    hadamard_factors: *const i32,
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
    #[specialize] a_per_token: bool,
) {
    assert_eq!(zero_points.is_some(), BITS == 4);
    // A reduction block holds 512 W4 or 256 W8 values, and the output rotation works on blocks of 32.
    let block_size = if BITS == 4 {
        512
    } else {
        256
    };
    assert!(in_vec_size.is_multiple_of(block_size));
    assert!(out_vec_size.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));
    let k_u = in_vec_size as usize;
    let n_u = out_vec_size as usize;
    let routes = num_routes as usize;
    let top_k = active_experts_per_token as usize;
    let bits = BITS as usize;
    let group_size = GROUP_SIZE as usize;

    for route_idx in 0..routes {
        let expert_id = unsafe { *expert_ids.add(route_idx) } as usize;
        let a_row_idx = if a_per_token {
            route_idx / top_k
        } else {
            route_idx
        };

        let expert_b = unsafe { b.add(expert_id * b_expert_stride_bytes as usize) };
        let expert_scales = unsafe { scales.add(expert_id * scale_expert_stride as usize) };
        let expert_zero_points =
            zero_points.map(|ptr| unsafe { ptr.add(expert_id * zero_point_expert_stride as usize) });
        let route_a = unsafe { a.add(a_row_idx * k_u) };
        let route_d = unsafe { d.add(route_idx * n_u) };
        let expert_hadamard = unsafe { hadamard_factors.add(expert_id * n_u) };

        let mut row_out = vec![0.0f32; n_u];
        for (output_dim_idx, out_slot) in row_out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for input_dim_idx in 0..k_u {
                let a_val = unsafe { *route_a.add(input_dim_idx) }.to_f32().unwrap();
                let weight_linear_index = output_dim_idx * k_u + input_dim_idx;
                let group_index = input_dim_idx / group_size;
                let metadata_index =
                    output_dim_idx * scale_output_stride as usize + group_index * scale_group_stride as usize;
                let scale = unsafe { *expert_scales.add(metadata_index) }.to_f32().unwrap();
                let zero_point = if BITS == 4 {
                    let zp_index = output_dim_idx * zero_point_output_stride as usize
                        + group_index * zero_point_group_stride as usize;
                    Some(unsafe { read_zero_point(expert_zero_points.unwrap(), zp_index, bits) })
                } else {
                    None
                };
                let b_val = unsafe {
                    dequantize_weight(
                        expert_b as *const u32,
                        weight_linear_index,
                        bits,
                        false,
                        scale,
                        zero_point,
                    )
                };
                acc += a_val * b_val;
            }
            *out_slot = acc;
        }

        let block_size = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
        for block_start in (0..n_u).step_by(block_size) {
            let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize];
            for lane in 0..block_size {
                stripe[lane] = T::from(row_out[block_start + lane]).unwrap().to_f32().unwrap();
            }
            hadamard_transform(&mut stripe);
            for lane in 0..block_size {
                let sign = unsafe { *expert_hadamard.add(block_start + lane) } as f32;
                let transformed = T::from(stripe[lane] * sign).unwrap().to_f32().unwrap();
                row_out[block_start + lane] = transformed;
            }
        }

        for (output_dim_idx, &val) in row_out.iter().enumerate() {
            unsafe {
                *route_d.add(output_dim_idx) = T::from(val).unwrap();
            }
        }
    }
}
