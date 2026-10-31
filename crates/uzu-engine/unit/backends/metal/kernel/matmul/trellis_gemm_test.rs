use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::kernel::{
            matmul::{MatmulShape, TrellisFormat},
            tests::trellis_test::projection::{check_projection, format, whole_blocks},
        },
        metal::{
            Metal,
            gpu_types::gemm::GemmTiling,
            kernel::matmul::{GemmKernel, MatmulOutputWork, gemm::GemmEngine},
        },
    },
    data_type::DataType,
    tests::util::shared_metal_context,
};

// Ordered as (m, n, k, tiling); every K is whole 64-column blocks.
const FORCED_PLANS: [(usize, usize, usize, GemmTiling); 5] = [
    (129, 132, 1024, GemmTiling::Tile128x128x256_Simdgroups4x4),
    (129, 132, 64, GemmTiling::Tile128x128x256_Simdgroups4x4),
    (17, 80, 192, GemmTiling::Tile32x64x256_Simdgroups2x2),
    (17, 80, 192, GemmTiling::Tile16x32x256_Simdgroups1x1),
    (7, 80, 192, GemmTiling::Tile16x32x256_Simdgroups1x1),
];

#[rstest]
#[test_attr(uzu_test)]
#[case::v4_t8(format(4, 8, Some(64)))]
#[case::v2_t6(format(2, 6, None))]
#[case::v2_t4(format(2, 4, None))]
#[case::v4_t6(format(4, 6, Some(64)))]
#[case::v4_t7(format(4, 7, Some(64)))]
#[case::v4_t6_r40(format(4, 6, Some(40)))]
#[case::v4_t8_no_restart(format(4, 8, None))]
fn trellis_forced_plans_match_tape_levels(#[case] format: TrellisFormat) {
    let context = shared_metal_context();
    if !context.supports_mxu || !whole_blocks(format, 64) {
        return;
    }
    for (m, n, k, tiling) in FORCED_PLANS {
        check_projection::<Metal>(&context, format, (m, n, k), |arguments, command_buffer| {
            let shape = MatmulShape::from_arguments(&arguments);
            let mut gemm = GemmKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
            let mut plan = gemm.select_plan_for_engine(&shape, GemmEngine::Mxu, &context).unwrap();
            plan.tiling = tiling;
            plan.split_k = 1;
            let output_work = MatmulOutputWork::new(&context, DataType::BF16, DataType::BF16).unwrap();
            gemm.encode_plan(arguments, plan, &output_work, command_buffer).unwrap();
        });
    }
}
