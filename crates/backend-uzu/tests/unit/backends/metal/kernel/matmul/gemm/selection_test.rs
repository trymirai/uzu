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
    GemmProblem::new(shape, data_type, data_type, data_type, true)
}

fn select(shape: MatmulShape) -> GemmPlan {
    select_plan(shape, DataType::BF16, DataType::BF16, true)
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
        assert_eq!(
            select_plan_for_engine(shape, DataType::BF16, DataType::BF16, true, Simdgroup).unwrap().tiling,
            tiling
        );
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
    assert!(should_stage_weight_scales(p, select(p)));

    p.b_group_size = Some(32);
    assert!(!should_stage_weight_scales(p, select(p)));

    p.d_transform = GemmDTransform::BIAS;
    assert_eq!(select_plan(p, DataType::BF16, DataType::F32, true).split_k, 1);

    let mut w8 = a8(quant(shape(17, 4096, 4096)));
    w8.b_bits = Some(8);
    assert_eq!(select(w8).split_k, 16);

    let mut groups = a8(quant(shape(32, 64, 320)));
    groups.b_bits = Some(8);
    let plan = plan(Mxu, Tile32x64x256_Simdgroups2x2, 1);
    assert!(!should_stage_weight_scales(groups, plan));
    groups.k = 384;
    assert!(should_stage_weight_scales(groups, plan));
}

#[uzu_test]
fn gemv_override_is_preserved() {
    let prefer = |shape, data_type| {
        let problem = problem(shape, data_type);
        prefer_gemm_over_gemv(problem, problem.select_plan())
    };

    assert!(prefer(shape(4, 4096, 8192), DataType::BF16));
    assert!(!prefer(shape(4, 8192, 4096), DataType::BF16));
    assert!(!prefer(shape(4, 4096, 4096), DataType::F32));
    assert!(!prefer(shape(3, 4096, 8192), DataType::BF16));
    assert!(!prefer(shape(5, 4096, 8192), DataType::BF16));

    let mut gathered = shape(4, 4096, 8192);
    gathered.gathered = true;
    assert!(!prefer(gathered, DataType::BF16));
}

#[uzu_test]
fn engine_tiling_mismatch_is_rejected() {
    let shape = quant(shape(64, 4096, 4096));
    assert!(matches!(
        validate_plan(
            shape,
            DataType::BF16,
            DataType::BF16,
            true,
            plan(GemmEngine::Simdgroup, GemmTiling::Tile64x64x256_Simdgroups2x2, 1)
        ),
        Err(GemmPlanError::EngineTilingMismatch { .. })
    ));
}
