use proc_macros::uzu_test;

use super::*;
use crate::backends::common::gpu_types::gemm::GemmDTransform;

fn shape(
    m: u32,
    n: u32,
    k: u32,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k,
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

fn quant(mut shape: MatmulShape) -> MatmulShape {
    shape.b_prologue = GemmBPrologueKind::ScaleSymmetricDequant;
    shape.b_bits = Some(4);
    shape.b_group_size = Some(64);
    shape.signed_codes = true;
    shape
}

fn problem(
    shape: MatmulShape,
    output_data_type: DataType,
) -> GemmProblem {
    GemmProblem::new(shape, DataType::BF16, output_data_type, true)
}

fn plan(
    engine: GemmEngine,
    tiling: GemmTiling,
    split_k: u32,
) -> GemmPlan {
    GemmPlan {
        engine,
        tiling,
        split_k,
    }
}

#[uzu_test]
fn policy_boundaries_are_preserved() {
    use GemmTiling::*;

    for (is_a_int8, m, n, expected) in [
        (true, 16, 4096, Tile16x128x256_Simdgroups1x4),
        (true, 17, 4096, Tile32x64x256_Simdgroups2x2),
        (true, 64, 63, Tile64x32x256_Simdgroups4x1),
        (true, 512, 4096, Tile128x128x256_Simdgroups4x4),
        (false, 16, 4096, Tile32x64x256_Simdgroups2x2),
        (false, 64, 4096, Tile64x64x256_Simdgroups2x2),
        (false, 256, 4096, Tile128x128x256_Simdgroups4x4),
    ] {
        assert_eq!(policy::mxu_mn_tile(is_a_int8, m, n), expected);
    }

    for (m, n, k, expected) in [
        (15, 2560, 2560, Tile16x32x256_Simdgroups1x1),
        (15, 4096, 8192, Tile16x128x256_Simdgroups1x4),
        (15, 131_073, 4096, Tile16x32x256_Simdgroups1x1),
        (64, 4096, 2048, Tile64x64x256_Simdgroups2x2),
        (256, 4096, 2048, Tile128x128x256_Simdgroups4x4),
    ] {
        assert_eq!(policy::mxu_fp_tile(m, n, k), expected);
    }

    for (m, n, group_size, expected) in [
        (16, 4096, 31, Tile64x64x16_Simdgroups2x2),
        (31, 4096, 32, Tile8x32x32_Simdgroups1x1),
        (64, 6143, 32, Tile32x32x32_Simdgroups2x2),
        (64, 6144, 32, Tile64x64x32_Simdgroups2x2),
    ] {
        assert_eq!(policy::simdgroup_quant_tile(m, n, group_size), expected);
    }
}

#[uzu_test]
fn selection_fallbacks_and_split_k_are_preserved() {
    use GemmEngine::*;
    use GemmTiling::*;

    let mut invalid_layout = quant(shape(64, 4096, 4096));
    invalid_layout.b_transpose = false;
    assert_eq!(problem(invalid_layout, DataType::BF16).select_plan().engine, Simdgroup);
    assert_eq!(problem(quant(shape(64, 4096, 4095)), DataType::BF16).select_plan().engine, Simdgroup);

    let mut large_group = quant(shape(512, 4096, 4096));
    large_group.a_full_precision = false;
    large_group.b_group_size = Some(128);
    assert_eq!(problem(large_group, DataType::BF16).select_plan().tiling, Tile64x64x256_Simdgroups2x2);

    let mut a8 = quant(shape(16, 4096, 4096));
    a8.a_full_precision = false;
    assert_eq!(problem(a8, DataType::BF16).select_plan().split_k, 8);

    let mut biased = quant(shape(16, 4096, 4096));
    biased.a_full_precision = false;
    biased.d_transform = GemmDTransform::BIAS;
    assert_eq!(GemmProblem::new(biased, DataType::BF16, DataType::F32, true).select_plan().split_k, 1);

    let mut zero = quant(shape(0, 1, 1));
    zero.b_prologue = GemmBPrologueKind::ScaleZeroPointDequant;
    zero.b_group_size = Some(u32::MAX);
    assert_eq!(problem(zero, DataType::BF16).select_plan().split_k, 1);
}

#[uzu_test]
fn invalid_plans_are_rejected() {
    let huge = shape(u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(
        GemmProblem::new(huge, DataType::BF16, DataType::BF16, false).select_plan_for_engine(GemmEngine::Mxu),
        Err(GemmPlanError::MxuUnavailable)
    );

    let mut invalid_layout = quant(huge);
    invalid_layout.b_transpose = false;
    assert_eq!(
        problem(invalid_layout, DataType::BF16).select_plan_for_engine(GemmEngine::Mxu),
        Err(GemmPlanError::UnsupportedQuantLayout)
    );

    let problem = problem(quant(shape(64, 4096, 4096)), DataType::BF16);
    assert!(matches!(
        problem.validate(plan(GemmEngine::Simdgroup, GemmTiling::Tile32x32x32_Simdgroups2x2, 3)),
        Err(GemmPlanError::InvalidSplitK {
            split_k: 3
        })
    ));
}
