use metal::MTLGPUFamily;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::common::{
        gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
        kernel::matmul::MatmulShape,
    },
    data_type::DataType,
};

const SMALL_GPU_CORE_COUNT: u32 = 29;
const LARGE_GPU_CORE_COUNT: u32 = 30;
const APPLE7: Option<MTLGPUFamily> = Some(MTLGPUFamily::Apple7);
const APPLE8: Option<MTLGPUFamily> = Some(MTLGPUFamily::Apple8);
const APPLE9: Option<MTLGPUFamily> = Some(MTLGPUFamily::Apple9);
const APPLE10: Option<MTLGPUFamily> = Some(MTLGPUFamily::Apple10);

fn qtile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
) -> GemvTile {
    GemvTile::quantized(num_simdgroups, results_per_simdgroup, 1, 32, 2)
}

#[uzu_test]
fn fp_policy_cases() {
    let cases = [
        (LARGE_GPU_CORE_COUNT, APPLE9, 1, 12288, 1536, true, tile(8, 8, 1)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 1, 12288, 1536, false, tile(8, 1, 1)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 1, 1536, 256, true, tile(8, 1, 1)),
        (SMALL_GPU_CORE_COUNT, APPLE9, 1, 1536, 256, true, tile(8, 2, 1)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 8, 12288, 1536, true, tile(8, 1, 4)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 1, 1536, 12288, true, tile(8, 8, 4)),
        (SMALL_GPU_CORE_COUNT, APPLE7, 1, 262144, 1536, true, tile(8, 1, 4)),
    ];

    for (gpu_core_count, apple_gpu_family, m, n, k, aligned, expected) in cases {
        assert_eq!(fp_tile(gpu_core_count, apple_gpu_family, m, n, k, aligned), expected);
    }
}

#[uzu_test]
fn quant_policy_cases() {
    let cases = [
        (LARGE_GPU_CORE_COUNT, APPLE9, 1, 256, 1536, 4, false, qtile(2, 1)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 1, 262144, 1536, 4, false, qtile(8, 4)),
        (SMALL_GPU_CORE_COUNT, APPLE9, 1, 1536, 256, 4, false, qtile(4, 4)),
        (SMALL_GPU_CORE_COUNT, APPLE10, 1, 1536, 256, 4, false, qtile(8, 4)),
        (SMALL_GPU_CORE_COUNT, APPLE8, 1, 2048, 1536, 4, false, qtile(8, 2)),
        (SMALL_GPU_CORE_COUNT, APPLE7, 1, 256, 1536, 4, false, qtile(8, 2)),
        (SMALL_GPU_CORE_COUNT, None, 1, 1536, 256, 4, false, qtile(4, 8)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 2, 2048, 1536, 4, false, qtile(8, 4)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 4, 5120, 5120, 4, false, qtile(8, 4)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 1, 2048, 1536, 8, false, GemvTile::quantized(8, 4, 1, 32, 4)),
        (LARGE_GPU_CORE_COUNT, APPLE9, 1, 2560, 9216, 4, true, qtile(4, 8)),
    ];

    for (gpu_core_count, apple_gpu_family, m, n, k, bits, has_rht, expected) in cases {
        let actual = quantized_tile(
            gpu_core_count,
            apple_gpu_family,
            bits,
            32,
            m,
            n,
            k,
            if has_rht {
                GemmDTransform::RHT
            } else {
                GemmDTransform::empty()
            },
            true,
        )
        .expect("quantized route");
        assert_eq!(
            actual, expected,
            "m={m} n={n} k={k} bits={bits} gpu_core_count={gpu_core_count} apple_gpu_family={apple_gpu_family:?}"
        );
    }
}

#[uzu_test]
fn gpu_core_count_boundary_selects_large_policy() {
    let transform = GemmDTransform::empty();
    let small = quantized(SMALL_GPU_CORE_COUNT, APPLE9, 4, 32, 1, 256, 1536, transform).expect("small GPU route");
    let large = quantized(LARGE_GPU_CORE_COUNT, APPLE9, 4, 32, 1, 256, 1536, transform).expect("large GPU route");

    assert_eq!(small, qtile(4, 2));
    assert_eq!(large, qtile(2, 1));
}

fn quantized(
    gpu_core_count: u32,
    apple_gpu_family: Option<MTLGPUFamily>,
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
    k: u32,
    d_transform: GemmDTransform,
) -> Option<GemvTile> {
    quantized_tile(gpu_core_count, apple_gpu_family, bits, group, m, n, k, d_transform, true)
}

#[uzu_test]
fn quantized_policy_edges() {
    for (gpu_core_count, apple_gpu_family, m, group, expected) in [
        (LARGE_GPU_CORE_COUNT, APPLE10, 2, 32, GemvTile::quantized(8, 4, 1, 32, 4)),
        (LARGE_GPU_CORE_COUNT, APPLE10, 3, 32, GemvTile::quantized(8, 4, 1, 32, 4)),
        (SMALL_GPU_CORE_COUNT, APPLE8, 4, 64, GemvTile::quantized(8, 4, 1, 32, 8)),
    ] {
        assert_eq!(
            quantized(gpu_core_count, apple_gpu_family, 8, group, m, 4096, 4096, GemmDTransform::RHT),
            Some(expected)
        );
    }
    let none = GemmDTransform::empty();
    assert_eq!(
        quantized(LARGE_GPU_CORE_COUNT, APPLE10, 4, 64, 4, 8192, 4100, none),
        Some(GemvTile::quantized(8, 4, 1, 32, 4))
    );
    assert!(quantized(LARGE_GPU_CORE_COUNT, APPLE10, 4, 32, 4, 8, 4096, none).is_some());
    assert!(quantized(LARGE_GPU_CORE_COUNT, APPLE10, 4, 32, 1, 8192, 4096, none).is_some());
    assert_eq!(quantized(LARGE_GPU_CORE_COUNT, APPLE10, 4, 32, 9, 8192, 4096, none), None);
    assert_eq!(
        quantized_tile(LARGE_GPU_CORE_COUNT, APPLE10, 4, 32, 4, 8192, 4096, none, true),
        Some(GemvTile::quantized(8, 4, 1, 32, 2))
    );
}

#[uzu_test]
fn untuned_quantized_io_uses_only_the_generated_fallback() {
    assert_eq!(quantized_tile(LARGE_GPU_CORE_COUNT, APPLE10, 4, 32, 4, 3, 4096, GemmDTransform::empty(), false,), None);
    let tile = quantized_tile(LARGE_GPU_CORE_COUNT, APPLE10, 4, 32, 4, 4096, 4096, GemmDTransform::empty(), false)
        .expect("mixed-IO fallback");
    assert_eq!(tile, GemvTile::quantized(8, 4, 1, 32, 2));
    assert_eq!(
        quantized_tile(LARGE_GPU_CORE_COUNT, APPLE10, 4, 32, 5, 4096, 4096, GemmDTransform::empty(), false,),
        None
    );
}

fn quant_shape(
    m: u32,
    n: u32,
    d_transform: GemmDTransform,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k: 4096,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::ScaleZeroPointDequant,
        b_bits: Some(4),
        b_group_size: Some(32),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        d_transform,
    }
}

#[uzu_test]
fn block_unaligned_quantized_k_stays_on_gemv() {
    let shape = MatmulShape {
        m: 1,
        n: 64,
        k: 1152,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::ScaleZeroPointDequant,
        b_bits: Some(4),
        b_group_size: Some(64),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        d_transform: GemmDTransform::empty(),
    };
    assert!(
        super::super::kernel::GemvSpecialization::select_shape(
            &shape,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            LARGE_GPU_CORE_COUNT,
            APPLE10,
        )
        .is_some()
    );
}

#[uzu_test]
fn specialization_preserves_quantized_route_and_accumulate_tail() {
    let select = |n, transform| {
        super::super::kernel::GemvSpecialization::select_shape(
            &quant_shape(4, n, transform),
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            LARGE_GPU_CORE_COUNT,
            APPLE10,
        )
    };
    let clean = select(8192, GemmDTransform::empty()).expect("quantized specialization");
    assert!(clean.output_row_tile() > DEFAULT_RESULTS_PER_SIMDGROUP);
    assert_eq!(select(8192 + clean.output_row_tile() / 2, GemmDTransform::ACCUMULATE), None);
}
