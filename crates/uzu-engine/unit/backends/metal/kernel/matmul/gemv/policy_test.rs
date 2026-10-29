use metal::MTLGPUFamily;
use uzu_engine_macros::uzu_test;

use super::{DEFAULT_RESULTS_PER_SIMDGROUP, GemvTile, fp_tile, quantized_tile, tile};
use crate::{
    backends::{
        common::kernel::matmul::{MatmulShape, QuantParamsLayout},
        metal::gpu_types::gemm::GemmBLayout,
    },
    data_type::DataType,
    tests::matmul::quantized_matmul_shape as shape,
};

const SMALL_GPU_CORE_COUNT: u32 = 29;
const LARGE_GPU_CORE_COUNT: u32 = 30;

fn qtile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
) -> GemvTile {
    GemvTile::quantized(num_simdgroups, results_per_simdgroup, 1, 32, 2)
}

#[uzu_test]
fn fp_policy_cases() {
    let cases = [
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 12288, 1536, true, tile(8, 8, 1)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 12288, 1536, false, tile(8, 1, 1)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 1536, 256, true, tile(8, 1, 1)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 1536, 256, true, tile(8, 2, 1)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 8, 12288, 1536, true, tile(8, 1, 4)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 1536, 12288, true, tile(8, 8, 4)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple7, 1, 262144, 1536, true, tile(8, 1, 4)),
    ];

    for (gpu_core_count, apple_gpu_family, m, n, k, aligned, expected) in cases {
        assert_eq!(fp_tile(gpu_core_count, apple_gpu_family, m, n, k, aligned), expected);
    }
}

#[uzu_test]
fn quant_policy_cases() {
    let cases = [
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 256, 1536, 4, qtile(2, 1)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 256, 1536, 4, qtile(4, 2)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 262144, 1536, 4, qtile(8, 4)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 1536, 256, 4, qtile(4, 4)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 1, 1536, 256, 4, qtile(8, 4)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple8, 1, 2048, 1536, 4, qtile(8, 2)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple7, 1, 256, 1536, 4, qtile(8, 2)),
        (SMALL_GPU_CORE_COUNT, MTLGPUFamily::Apple7, 1, 1536, 256, 4, qtile(4, 8)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 2, 2048, 1536, 4, qtile(8, 4)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 4, 5120, 5120, 4, qtile(8, 4)),
        (LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple9, 1, 2048, 1536, 8, GemvTile::quantized(8, 4, 1, 32, 4)),
    ];

    for (gpu_core_count, apple_gpu_family, m, n, k, bits, expected) in cases {
        let actual =
            quantized_tile(gpu_core_count, apple_gpu_family, bits, 32, m, n, k, true).expect("quantized route");
        assert_eq!(
            actual, expected,
            "m={m} n={n} k={k} bits={bits} gpu_core_count={gpu_core_count} apple_gpu_family={apple_gpu_family:?}"
        );
    }
}

#[uzu_test]
fn quantized_policy_edges() {
    assert_eq!(
        quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 64, 4, 8192, 4100, true),
        Some(GemvTile::quantized(8, 4, 1, 32, 4))
    );
    assert!(quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 4, 8, 4096, true).is_some());
    assert!(quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 1, 8192, 4096, true).is_some());
    assert_eq!(quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 9, 8192, 4096, true), None);
    assert_eq!(
        quantized_tile(LARGE_GPU_CORE_COUNT, MTLGPUFamily::Apple10, 4, 32, 4, 8192, 4096, true),
        Some(GemvTile::quantized(8, 4, 1, 32, 2))
    );
}

#[uzu_test]
fn block_unaligned_quantized_k_stays_on_gemv() {
    let shape = MatmulShape {
        params_layout: Some(QuantParamsLayout::OutputGroup),
        ..shape(1, 64, 1152, 4, 64)
    };
    assert!(
        super::super::kernel::GemvSpecialization::select_shape(
            &shape,
            GemmBLayout::RowMajor,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            LARGE_GPU_CORE_COUNT,
            MTLGPUFamily::Apple10,
        )
        .is_some()
    );
}

#[uzu_test]
fn specialization_preserves_quantized_route() {
    let select = |n| {
        super::super::kernel::GemvSpecialization::select_shape(
            &MatmulShape {
                params_layout: Some(QuantParamsLayout::OutputGroup),
                ..shape(4, n, 4096, 4, 32)
            },
            GemmBLayout::RowMajor,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            LARGE_GPU_CORE_COUNT,
            MTLGPUFamily::Apple10,
        )
    };
    let clean = select(8192).expect("quantized specialization");
    assert!(clean.output_row_tile() > DEFAULT_RESULTS_PER_SIMDGROUP);
    assert!(select(8192 + clean.output_row_tile() / 2).is_some());
}

#[uzu_test]
fn gathered_group_major_is_a_gemv_route() {
    let mut shape = shape(1, 8192, 4096, 4, 32);
    shape.gathered = true;
    assert!(
        super::super::kernel::GemvSpecialization::select_tile(&shape, GemmBLayout::RowMajor, qtile(8, 4),).is_some()
    );
}
