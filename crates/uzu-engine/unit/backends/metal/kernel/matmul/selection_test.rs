use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::common::{
        gpu_types::gemm::GemmDTransform,
        kernel::{ActivationQuantization, matmul::Int8CodeLayout},
    },
    tests::matmul::{matmul_shape, quantized_matmul_shape as shape},
};

fn selector(supports_mxu: bool) -> MatmulSelector {
    MatmulSelector {
        gpu_core_count: 40,
        apple_gpu_family: if supports_mxu {
            MTLGPUFamily::Apple10
        } else {
            MTLGPUFamily::Apple9
        },
        supports_mxu,
        data_types: (DataType::BF16, DataType::BF16, DataType::BF16),
    }
}

#[uzu_test]
fn gemv_gemm_route_boundaries_are_preserved() {
    let selector = selector(true);
    for (shape, prefer_gemm) in [
        (matmul_shape(4, 4096, 8192), true),
        (matmul_shape(4, 8192, 4096), false),
        (matmul_shape(3, 4096, 8192), false),
        (matmul_shape(5, 4096, 8192), false),
    ] {
        let plan = selector.problem(shape).select_plan();
        assert_eq!(selector.prefer_gemm_over_gemv(shape, plan), prefer_gemm);
    }
    let mut gathered = matmul_shape(4, 4096, 8192);
    gathered.gathered = true;
    assert!(!selector.prefer_gemm_over_gemv(gathered, selector.problem(gathered).select_plan()));
}

#[uzu_test]
fn qmv_route_boundaries() {
    use GemmBLayout::{MxuTiled, RowMajor};

    for (supports_mxu, layout, m, n, k, bits, group, gemm_route, soft_cap) in [
        (false, RowMajor, 2, 72, 1024, 4, 64, false, false),
        (false, RowMajor, 8, 72, 1024, 4, 64, false, false),
        (true, RowMajor, 5, 72, 1024, 4, 32, false, false),
        (false, RowMajor, 4, 72, 1024, 4, 64, true, false),
        (false, RowMajor, 4, 72, 256, 8, 64, true, false),
        (true, RowMajor, 4, 72, 1024, 4, 64, false, false),
        (true, RowMajor, 5, 72, 1024, 4, 32, true, false),
        (true, RowMajor, 5, 72, 768, 8, 64, true, false),
        (true, RowMajor, 3, 72, 1024, 4, 32, false, false),
        (true, RowMajor, 8, 72, 1024, 8, 64, true, true),
        (true, MxuTiled, 5, 64, 1024, 4, 64, false, false),
        (true, MxuTiled, 6, 64, 1024, 4, 64, false, false),
        (true, MxuTiled, 8, 64, 128, 4, 64, false, false),
        (true, MxuTiled, 8, 64, 128, 8, 64, false, false),
    ] {
        let mut problem = shape(m, n, k, bits, group);
        if bits == 4 && gemm_route {
            problem.a = ActivationFormat::PackedHalf {
                group_size: group,
            };
        }
        problem.d_transform.set(GemmDTransform::SOFT_CAP, soft_cap);
        let actual = selector(supports_mxu).select(&problem, layout, ActivationFormat::Bf16).dispatch;
        if gemm_route {
            let expected_tiling = if bits == 4 {
                GemmTiling::Tile8x32x64_Simdgroups1x2x4
            } else {
                GemmTiling::Tile8x32x128_Simdgroups1x2x4
            };
            assert!(
                matches!(actual, MatmulDispatch::Gemm(plan) if plan.tiling == expected_tiling),
                "M={m} W{bits} G{group}, MXU={supports_mxu}"
            );
        } else {
            assert!(matches!(actual, MatmulDispatch::Gemv(_)), "M={m} W{bits} G{group}, MXU={supports_mxu}");
        }
    }
}

#[uzu_test]
fn int8_preparation_requires_tileable_weights() {
    for (n, k, bits, expected) in [
        (96, 256, 4, true),
        (72, 256, 4, false),
        (64, 40960, 4, true),
        (64, 41088, 4, false),
        (64, 20480, 8, true),
        (64, 20608, 8, false),
    ] {
        assert_eq!(
            matches!(
                selector(true).activation_preparation(&shape(16, n, k, bits, 64), GemmBLayout::RowMajor),
                ActivationFormat::Int8(_)
            ),
            expected,
            "n={n} k={k} W{bits}"
        );
    }
}

#[uzu_test]
fn selection_uses_only_prepared_formats() {
    use GemmBLayout::{MxuTiled, RowMajor};
    let packed = ActivationFormat::PackedHalf {
        group_size: 64,
    };
    let int8 =
        ActivationFormat::Int8(ActivationQuantization::new(128, 64, false, Int8CodeLayout::GroupedByNibble).unwrap());
    for (mxu, m, n, k, layout, prepared, expected) in [
        (false, 4, 5120, 6144, RowMajor, packed, packed),
        (false, 512, 5120, 6144, RowMajor, packed, ActivationFormat::Bf16),
        (true, 4, 5120, 6144, MxuTiled, int8, ActivationFormat::Bf16),
        (true, 5, 5120, 6144, MxuTiled, int8, int8),
        (true, 8, 5120, 6144, MxuTiled, int8, int8),
        (true, 512, 5120, 6144, MxuTiled, int8, int8),
        (true, 5, 72, 512, RowMajor, packed, packed),
    ] {
        let selector = selector(mxu);
        let problem = shape(m, n, k, 4, 64);
        assert_eq!(selector.activation_preparation(&problem, layout), prepared);
        let selected = selector.select(&problem, layout, prepared);
        assert_eq!(selected.activation_format, expected, "MXU={mxu} M={m} N={n} K={k}");
        assert_eq!(selector.select(&problem, layout, ActivationFormat::Bf16).activation_format, ActivationFormat::Bf16);
        match selected.activation_format {
            ActivationFormat::Int8(_) => {
                assert!(matches!(selected.dispatch, MatmulDispatch::Gemm(plan) if plan.engine == GemmEngine::Mxu))
            },
            ActivationFormat::PackedHalf {
                ..
            } => assert!(matches!(selected.dispatch, MatmulDispatch::Gemm(plan) if plan.tiling.is_packed_w4())),
            _ => {},
        }
    }
}
