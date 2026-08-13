use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::{
        common::gpu_types::{ActivationType, GatedActMulOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
        cpu::kernel::activation_transform::{hadamard_transform, quantize_transformed_row},
    },
};

#[kernel(GatedActMul)]
#[variants(T, f32, bf16)]
pub fn gated_act_mul<T: ArrayElement + Float>(
    act_operand: *const T,
    #[optional(!interleaved)] value_operand: Option<*const T>,
    #[optional(ops == GatedActMulOp::FullPrecision)] fp_out: Option<*mut T>,
    #[optional(ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums)] q_out: Option<*mut i8>,
    #[optional(ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums)] scales_out: Option<
        *mut f32,
    >,
    #[optional(ops == GatedActMulOp::QuantizeWithGroupSums)] group_sums_out: Option<*mut i32>,
    #[optional(use_hadamard)] hadamard_factors: Option<*const i32>,
    gated_dim: u32,
    batch_dim: u32,
    value_offset: u32,
    value_row_stride: u32,
    act_type: ActivationType,
    #[specialize] ops: GatedActMulOp,
    #[specialize] interleaved: bool,
    #[specialize] use_hadamard: bool,
    #[specialize] activation_scale_group_size: u32,
    #[specialize] sum_group_size: u32,
) {
    assert_eq!(hadamard_factors.is_some(), use_hadamard);
    let quantize = matches!(ops, GatedActMulOp::Quantize | GatedActMulOp::QuantizeWithGroupSums);
    let emit_group_sums = ops == GatedActMulOp::QuantizeWithGroupSums;
    assert!(!quantize || use_hadamard, "quantized gate activation requires RHT");
    assert!(!quantize || gated_dim.is_multiple_of(activation_scale_group_size));
    assert!(!emit_group_sums || gated_dim.is_multiple_of(sum_group_size));

    let gated_dim = gated_dim as usize;
    let batch_dim = batch_dim as usize;
    let activation_scale_group_size = activation_scale_group_size as usize;
    let sum_group_size = sum_group_size as usize;
    assert!(!use_hadamard || gated_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));

    for batch in 0..batch_dim {
        let mut transformed = use_hadamard.then(|| vec![0.0f32; gated_dim]);
        for gated in 0..gated_dim {
            let (act_index, value) = if interleaved {
                let base = batch * 2 * gated_dim;
                (base + gated_dim + gated, unsafe { *act_operand.add(base + gated) })
            } else {
                let value_index = batch * value_row_stride as usize + value_offset as usize + gated;
                (batch * gated_dim + gated, unsafe { *value_operand.unwrap().add(value_index) })
            };
            let gate = unsafe { *act_operand.add(act_index) };
            let result = super::gated_act_mul(value, gate, act_type);
            if let Some(transformed) = transformed.as_mut() {
                transformed[gated] = result;
            } else {
                unsafe {
                    *fp_out.expect("FP gate activation requires fp_out").add(batch * gated_dim + gated) =
                        T::from(result).unwrap();
                }
            }
        }

        if let Some(transformed) = transformed.as_mut() {
            let factors = hadamard_factors.expect("Hadamard factors are required");
            for stripe_start in (0..gated_dim).step_by(HADAMARD_TRANSFORM_BLOCK_SIZE) {
                let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE];
                for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE {
                    let gated = stripe_start + lane;
                    stripe[lane] = transformed[gated] * unsafe { *factors.add(gated) } as f32;
                }
                hadamard_transform(&mut stripe);
                transformed[stripe_start..stripe_start + HADAMARD_TRANSFORM_BLOCK_SIZE].copy_from_slice(&stripe);
            }

            if quantize {
                let values = unsafe {
                    std::slice::from_raw_parts_mut(
                        q_out.expect("quantized gate activation requires q_out").add(batch * gated_dim),
                        gated_dim,
                    )
                };
                let scales_per_row = gated_dim / activation_scale_group_size;
                let scales = unsafe {
                    std::slice::from_raw_parts_mut(
                        scales_out.expect("quantized gate activation requires scales_out").add(batch * scales_per_row),
                        scales_per_row,
                    )
                };
                let group_sums = group_sums_out.map(|output| unsafe {
                    std::slice::from_raw_parts_mut(
                        output.add(batch * gated_dim / sum_group_size),
                        gated_dim / sum_group_size,
                    )
                });
                quantize_transformed_row(
                    transformed,
                    activation_scale_group_size,
                    emit_group_sums.then_some(sum_group_size),
                    values,
                    scales,
                    group_sums,
                );
            } else {
                let output = fp_out.expect("FP gate activation requires fp_out");
                for (gated, &value) in transformed.iter().enumerate() {
                    unsafe { *output.add(batch * gated_dim + gated) = T::from(value).unwrap() };
                }
            }
        }
    }
}
