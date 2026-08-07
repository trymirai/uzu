use proc_macros::uzu_test;

use super::*;
use crate::backends::common::gpu_types::gemm::{GemmBPrologueKind, GemmDTransform};

fn shape() -> MatmulShape {
    MatmulShape {
        m: 64,
        n: 4096,
        k: 4096,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::FullPrecision,
        b_bits: None,
        b_group_size: None,
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        d_transform: GemmDTransform::empty(),
    }
}

fn plan(split_k: u32) -> GemmPlan {
    GemmPlan {
        engine: GemmEngine::Mxu,
        tiling: GemmTiling::Tile64x64x256_Simdgroups2x2,
        split_k,
    }
}

fn specialization(
    shape: MatmulShape,
    plan: GemmPlan,
) -> GemmSpecialization {
    GemmSpecialization::from_plan(
        plan,
        shape,
        DataType::BF16,
        GemmDTransform::empty(),
        GemmAlignment::new(true, true, true),
        GemmAPrologueKind::FullPrecision,
        None,
    )
    .unwrap()
}

#[uzu_test]
fn plan_flags_preserve_dense_and_split_k_defaults() {
    let dense = shape();
    let unsplit = specialization(dense, plan(1));
    assert!(unsplit.stage_weight_scales);
    assert!(!unsplit.hoist_operand_addressing);

    let split = specialization(dense, plan(2));
    assert!(split.stage_weight_scales);
    assert!(split.hoist_operand_addressing);
}

#[uzu_test]
fn plan_flags_use_quantized_tuning() {
    let mut quant = shape();
    quant.b_prologue = GemmBPrologueKind::ScaleSymmetricDequant;
    quant.b_bits = Some(4);
    quant.b_group_size = Some(64);
    quant.signed_codes = true;

    let spec = specialization(quant, plan(1));
    assert!(spec.stage_weight_scales);
    assert!(spec.hoist_operand_addressing);
}
