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

fn a8(mut shape: MatmulShape) -> MatmulShape {
    shape.a_full_precision = false;
    shape
}

fn problem(
    shape: MatmulShape,
    data_type: DataType,
) -> GemmProblem {
    GemmProblem::new(shape, data_type, data_type, true)
}

fn select(shape: MatmulShape) -> GemmPlan {
    problem(shape, DataType::BF16).select_plan()
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
fn tiling_boundaries_are_preserved() {
    use GemmEngine::*;
    use GemmTiling::*;

    let a8_cases = [
        (16, 4096, Tile16x128x256_Simdgroups1x4, 8),
        (17, 4096, Tile32x64x256_Simdgroups2x2, 8),
        (63, 4096, Tile32x64x256_Simdgroups2x2, 4),
        (64, 4096, Tile64x64x256_Simdgroups2x2, 4),
        (511, 4096, Tile64x64x256_Simdgroups2x2, 1),
        (512, 4096, Tile128x128x256_Simdgroups4x4, 2),
        (64, 63, Tile64x32x256_Simdgroups4x1, 16),
        (64, 64, Tile64x64x256_Simdgroups2x2, 16),
    ];
    for (m, n, tiling, split_k) in a8_cases {
        assert_eq!(select(a8(quant(shape(m, n, 4096)))), plan(Mxu, tiling, split_k));
    }

    let dense_cases = [
        (15, 2560, 2560, Tile16x32x256_Simdgroups1x1),
        (16, 2560, 2560, Tile32x64x256_Simdgroups2x2),
        (15, 4096, 8192, Tile16x128x256_Simdgroups1x4),
        (15, 131_073, 4096, Tile16x32x256_Simdgroups1x1),
        (15, 16_384, 4096, Tile16x128x256_Simdgroups1x4),
        (15, 8192, 4096, Tile32x64x256_Simdgroups2x2),
        (63, 4096, 2048, Tile32x64x256_Simdgroups2x2),
        (64, 4096, 2048, Tile64x64x256_Simdgroups2x2),
        (255, 4096, 2048, Tile64x64x256_Simdgroups2x2),
        (256, 4096, 2048, Tile128x128x256_Simdgroups4x4),
    ];
    for (m, n, k, tiling) in dense_cases {
        assert_eq!(select(shape(m, n, k)).tiling, tiling);
    }

    let simdgroup_cases = [
        (16, 4096, 31, Tile64x64x16_Simdgroups2x2),
        (31, 4096, 32, Tile8x32x32_Simdgroups1x1),
        (32, 4096, 32, Tile32x32x32_Simdgroups2x2),
        (64, 6143, 32, Tile32x32x32_Simdgroups2x2),
        (64, 6144, 32, Tile64x64x32_Simdgroups2x2),
    ];
    for (m, n, group_size, tiling) in simdgroup_cases {
        let mut shape = quant(shape(m, n, 8192));
        shape.b_group_size = Some(group_size);
        assert_eq!(problem(shape, DataType::BF16).select_plan_for_engine(Simdgroup).unwrap().tiling, tiling);
    }

    let mut invalid_layout = quant(shape(64, 4096, 4096));
    invalid_layout.b_transpose = false;
    assert_eq!(select(invalid_layout).engine, Simdgroup);
    assert_eq!(select(quant(shape(64, 4096, 4095))).engine, Simdgroup);

    let mut large_group = a8(quant(shape(512, 4096, 4096)));
    large_group.b_group_size = Some(128);
    assert_eq!(select(large_group).tiling, Tile64x64x256_Simdgroups2x2);
}

#[uzu_test]
fn split_k_and_staging_rules_are_preserved() {
    use GemmEngine::Mxu;
    use GemmTiling::Tile32x64x256_Simdgroups2x2;

    let mut p = a8(quant(shape(16, 4096, 4096)));
    assert!(select(p).should_stage_weight_scales(p));

    p.b_group_size = Some(32);
    assert!(!select(p).should_stage_weight_scales(p));

    p.d_transform = GemmDTransform::BIAS;
    assert_eq!(GemmProblem::new(p, DataType::BF16, DataType::F32, true).select_plan().split_k, 1);

    let mut w8 = a8(quant(shape(17, 4096, 4096)));
    w8.b_bits = Some(8);
    assert_eq!(select(w8).split_k, 16);

    let mut groups = a8(quant(shape(32, 64, 320)));
    groups.b_bits = Some(8);
    let plan = plan(Mxu, Tile32x64x256_Simdgroups2x2, 1);
    assert!(!plan.should_stage_weight_scales(groups));
    groups.k = 384;
    assert!(plan.should_stage_weight_scales(groups));

    let mut zero = quant(shape(0, 1, 1));
    zero.b_prologue = GemmBPrologueKind::ScaleZeroPointDequant;
    zero.b_group_size = Some(u32::MAX);
    assert_eq!(select(zero).split_k, 1);
}

#[uzu_test]
fn invalid_plans_are_rejected() {
    let huge = shape(u32::MAX, u32::MAX, u32::MAX);
    let no_mxu = GemmProblem::new(huge, DataType::BF16, DataType::BF16, false);
    assert_eq!(no_mxu.select_plan_for_engine(GemmEngine::Mxu), Err(GemmPlanError::MxuUnavailable));

    let mut invalid_layout = quant(huge);
    invalid_layout.b_transpose = false;
    assert_eq!(
        problem(invalid_layout, DataType::BF16).select_plan_for_engine(GemmEngine::Mxu),
        Err(GemmPlanError::UnsupportedQuantLayout)
    );

    let shape = quant(shape(64, 4096, 4096));
    assert!(matches!(
        problem(shape, DataType::BF16)
            .validate(plan(GemmEngine::Simdgroup, GemmTiling::Tile32x32x32_Simdgroups2x2, 3,)),
        Err(GemmPlanError::InvalidSplitK {
            split_k: 3
        })
    ));
}
