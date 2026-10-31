use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::kernel::matmul::{MatmulShape, TrellisFormat},
        metal::{
            Metal,
            gpu_types::gemm::GemmTiling,
            kernel::matmul::{GemmKernel, MatmulOutputWork, gemm::GemmEngine},
        },
    },
    data_type::DataType,
    tests::{
        matmul::trellis::{check_projection, format, whole_blocks},
        util::shared_metal_context,
    },
};

// Ordered as (m, n, k, tiling, split_k); every K is whole 64-column blocks.
const FORCED_PLANS: [(usize, usize, usize, GemmTiling, u32); 12] = [
    (1, 1, 512, GemmTiling::Tile16x32x32_Simdgroups2x1, 1),
    (1, 1, 512, GemmTiling::Tile16x32x256_Simdgroups1x1, 1),
    (1, 1, 2048, GemmTiling::Tile16x32x32_Simdgroups2x1, 1),
    (1, 1, 2048, GemmTiling::Tile16x32x256_Simdgroups1x1, 1),
    (129, 132, 1024, GemmTiling::Tile128x128x256_Simdgroups4x4, 1),
    (129, 132, 64, GemmTiling::Tile128x128x256_Simdgroups4x4, 1),
    (17, 80, 192, GemmTiling::Tile32x64x256_Simdgroups2x2, 1),
    (17, 80, 192, GemmTiling::Tile16x32x256_Simdgroups1x1, 1),
    (7, 80, 192, GemmTiling::Tile16x32x256_Simdgroups1x1, 1),
    (9, 68, 6144, GemmTiling::Tile16x32x32_Simdgroups2x1, 4),
    (33, 68, 6144, GemmTiling::Tile32x32x32_Simdgroups2x2, 4),
    (65, 68, 6144, GemmTiling::Tile64x64x32_Simdgroups2x2, 4),
];

fn check_forced_plan(
    format: TrellisFormat,
    dimensions: (usize, usize, usize),
    tiling: GemmTiling,
    split_k: u32,
) {
    let context = shared_metal_context();
    check_projection::<Metal>(&context, format, dimensions, |arguments, command_buffer| {
        let shape = MatmulShape::from_arguments(&arguments);
        let mut gemm = GemmKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
        let engine = if tiling.is_mxu_variant() {
            GemmEngine::Mxu
        } else {
            GemmEngine::Simdgroup
        };
        let mut plan = gemm.select_plan_for_engine(&shape, engine, &context).unwrap();
        plan.tiling = tiling;
        plan.split_k = split_k;
        let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();
        gemm.encode_plan(arguments, plan, &output_work, command_buffer).unwrap();
    });
}

#[rstest]
#[test_attr(uzu_test)]
#[case::v4_t8(format(4, 8, Some(64)))]
#[case::v2_t6(format(2, 6, None))]
#[case::v2_t4(format(2, 4, None))]
#[case::v4_t6(format(4, 6, Some(64)))]
#[case::v4_t7(format(4, 7, Some(64)))]
#[case::v4_t6_r40(format(4, 6, Some(40)))]
#[case::v4_t8_no_restart(format(4, 8, None))]
#[case::v2_t1(format(2, 1, None))]
#[case::v4_t1(format(4, 1, None))]
#[case::v4_t2(format(4, 2, None))]
fn trellis_forced_plans_match_tape_levels(#[case] format: TrellisFormat) {
    let supports_mxu = shared_metal_context().supports_mxu;
    // Simdgroup plans with m up to 8 run the decode kernel. K fits every restart size.
    for m in 1..=8 {
        check_forced_plan(format, (m, 37, 4160), GemmTiling::Tile16x32x32_Simdgroups2x1, 1);
    }
    // The forced plans' K are whole 64-column blocks.
    if !whole_blocks(format, 64) {
        return;
    }
    for (m, n, k, tiling, split_k) in FORCED_PLANS {
        if supports_mxu || !tiling.is_mxu_variant() {
            check_forced_plan(format, (m, n, k), tiling, split_k);
        }
    }
}
