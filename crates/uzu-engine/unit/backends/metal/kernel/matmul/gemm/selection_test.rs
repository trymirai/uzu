use metal::MTLGPUFamily;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::backends::{
    common::{gpu_types::gemm::GemmDTransform, kernel::matmul::QuantParamsLayout},
    metal::kernel::matmul::MatmulMetalKernel,
};

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
        b_is_trellis: false,
        b_bits: None,
        b_group_size: None,
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        params_layout: Some(QuantParamsLayout::OutputGroup),
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
    GemmProblem::new(shape, DataType::BF16, output_data_type, true, MTLGPUFamily::Apple7)
}

#[uzu_test]
fn policy_boundaries_are_preserved() {
    use GemmTiling::*;

    for (is_a_int8, m, n, expected) in [
        (true, 16, 4096, Tile16x32x256_Simdgroups1x1),
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

    for (m, n, family, expected) in [
        (16, 4096, MTLGPUFamily::Apple8, Tile8x32x32_Simdgroups1x1),
        (8, 4096, MTLGPUFamily::Apple8, Tile8x32x32_Simdgroups1x1),
        (9, 4096, MTLGPUFamily::Apple8, Tile32x32x32_Simdgroups2x2),
        (9, 4096, MTLGPUFamily::Apple9, Tile8x32x32_Simdgroups1x1),
        (31, 4096, MTLGPUFamily::Apple9, Tile8x32x32_Simdgroups1x1),
        (32, 4096, MTLGPUFamily::Apple9, Tile32x32x32_Simdgroups2x2),
        (64, 6143, MTLGPUFamily::Apple9, Tile32x32x32_Simdgroups2x2),
        (64, 6144, MTLGPUFamily::Apple9, Tile64x64x32_Simdgroups2x2),
    ] {
        assert_eq!(policy::simdgroup_quant_tile(m, n, family), expected);
    }
}

#[uzu_test]
fn selection_fallbacks_and_split_k_are_preserved() {
    use GemmEngine::*;
    use GemmTiling::*;

    for (mut case, expect_split) in [(shape(4, 32, 1024), true), (shape(64, 128, 128), false)] {
        case.d_transform = GemmDTransform::RHT | GemmDTransform::BIAS;
        for engine in [Simdgroup, Mxu] {
            let plan = problem(case, DataType::BF16)
                .select_plan_for_engine(engine)
                .expect("parity anchor supports the forced GEMM engine");
            assert_eq!(plan.split_k > 1, expect_split, "forced {engine:?} RHT+bias parity anchor");
        }
    }

    let mut non_split_quant_rht_bias = shape(9, 64, 64);
    non_split_quant_rht_bias.b_prologue = GemmBPrologueKind::ScaleZeroPointDequant;
    non_split_quant_rht_bias.b_bits = Some(4);
    non_split_quant_rht_bias.b_group_size = Some(64);
    non_split_quant_rht_bias.params_layout = Some(QuantParamsLayout::GroupOutput);
    non_split_quant_rht_bias.d_transform = GemmDTransform::RHT | GemmDTransform::BIAS;
    let plan = problem(non_split_quant_rht_bias, DataType::BF16).select_plan();
    assert_eq!(plan.engine, Simdgroup);
    assert_eq!(plan.split_k, 1);

    let mut invalid_layout = quant(shape(64, 4096, 4096));
    invalid_layout.b_transpose = false;
    assert_eq!(problem(invalid_layout, DataType::BF16).select_plan().engine, Simdgroup);
    invalid_layout.a_full_precision = false;
    assert_eq!(problem(invalid_layout, DataType::BF16).select_plan().engine, Mxu);
    assert_eq!(problem(quant(shape(64, 4096, 4095)), DataType::BF16).select_plan().engine, Simdgroup);

    for (m, n, k, expected_tiling, expected_split_k) in [
        (16, 4096, 4096, Tile16x32x256_Simdgroups1x1, 8),
        (16, 1024, 4096, Tile16x32x256_Simdgroups1x1, 16),
        (16, 34816, 4096, Tile16x32x256_Simdgroups1x1, 1),
    ] {
        let mut a8 = quant(shape(m, n, k));
        a8.a_full_precision = false;
        let plan = problem(a8, DataType::BF16).select_plan();
        assert_eq!(plan.tiling, expected_tiling);
        assert_eq!(plan.split_k, expected_split_k);
    }

    let mut biased = quant(shape(16, 4096, 4096));
    biased.a_full_precision = false;
    biased.d_transform = GemmDTransform::BIAS;
    assert_eq!(
        GemmProblem::new(biased, DataType::BF16, DataType::F32, true, MTLGPUFamily::Apple7).select_plan().split_k,
        1
    );

    let mut zero = quant(shape(0, 1, 1));
    zero.b_prologue = GemmBPrologueKind::ScaleZeroPointDequant;
    zero.b_group_size = Some(u32::MAX);
    assert_eq!(problem(zero, DataType::BF16).select_plan().split_k, 1);
}

#[uzu_test]
fn trellis_plan_matches_projection_cases() {
    use GemmTiling::*;

    for (m, n, k, tiling, split_k) in [
        (16, 128, 64, Tile16x32x256_Simdgroups1x1, 1),
        (17, 80, 64, Tile64x64x256_Simdgroups2x2, 1),
        (2048, 2048, 128, Tile128x128x256_Simdgroups4x4, 1),
        (1, 80, 5120, Tile16x32x256_Simdgroups1x1, 80),
        (1, 6, 5120, Tile64x64x256_Simdgroups2x2, 1),
    ] {
        let mut trellis_shape = shape(m, n, k);
        trellis_shape.a_full_precision = false;
        trellis_shape.b_is_trellis = true;
        let plan = problem(trellis_shape, DataType::BF16).select_plan();
        assert_eq!((plan.engine, plan.tiling, plan.split_k), (GemmEngine::Mxu, tiling, split_k), "M {m} N {n} K {k}");
    }
}

#[uzu_test]
fn forced_engine_errors_are_preserved() {
    let huge = shape(u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(
        GemmProblem::new(huge, DataType::BF16, DataType::BF16, false, MTLGPUFamily::Apple7)
            .select_plan_for_engine(GemmEngine::Mxu),
        Err(GemmPlanError::MxuUnavailable)
    );

    let mut trellis = huge;
    trellis.b_is_trellis = true;
    for mut packed in [quant(huge), trellis] {
        packed.b_transpose = false;
        assert_eq!(
            problem(packed, DataType::BF16).select_plan_for_engine(GemmEngine::Mxu),
            Err(GemmPlanError::UnsupportedLayout("packed weights require transposed contiguous B"))
        );
    }
}

#[uzu_test]
fn gemv_gemm_route_boundaries_are_preserved() {
    for (shape, data_type, prefer_gemm) in [
        (shape(4, 4096, 8192), DataType::BF16, true),
        (shape(4, 8192, 4096), DataType::BF16, false),
        (shape(4, 4096, 4096), DataType::F32, false),
        (shape(3, 4096, 8192), DataType::BF16, false),
        (shape(5, 4096, 8192), DataType::BF16, false),
    ] {
        let plan = GemmProblem::new(shape, data_type, data_type, true, MTLGPUFamily::Apple7).select_plan();
        assert_eq!(MatmulMetalKernel::prefer_gemm_over_gemv(shape, plan, data_type, data_type, data_type), prefer_gemm);
    }
    let mut gathered = shape(4, 4096, 8192);
    gathered.gathered = true;
    let plan = problem(gathered, DataType::BF16).select_plan();
    assert!(!MatmulMetalKernel::prefer_gemm_over_gemv(gathered, plan, DataType::BF16, DataType::BF16, DataType::BF16,));
}
