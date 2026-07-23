use crate::backends::{
    common::{
        Kernels,
        gpu_types::gemm::{gemm_tiling_simdgroups_per_column, gemm_tiling_simdgroups_per_row},
    },
    metal::Metal,
};

pub mod attention;
pub mod gdn;
pub mod matmul;
mod radix_top_k_small;

pub const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

include!(concat!(env!("OUT_DIR"), "/dsl.rs"));

pub struct MetalKernels;

impl Kernels for MetalKernels {
    type Backend = Metal;

    autogen_kernels!();
    type AttentionGemmCore = attention::AttentionGemmMetalCore;
    type DeltaNetChunkedPrefill = gdn::chunked::MetalDeltaNetChunkedPrefill;
    type DeltaNetTreeVerify = gdn::tree_verify::MetalDeltaNetTreeVerify;
    type MatmulKernel = matmul::MatmulMetalKernel;
    type RadixTopKSmall = radix_top_k_small::MetalRadixTopKSmall;
}

#[cfg(test)]
mod generated_request_tests {
    use super::{AttentionGemmRequest, GemmRequest, GemvRequest};
    use crate::{
        backends::{
            common::gpu_types::gemm::{GemmBPrologueKind, GemmTiling},
            metal::error::MetalError,
        },
        data_type::DataType,
    };

    fn full_precision_gemm(tiling: GemmTiling) -> GemmRequest {
        GemmRequest {
            at: DataType::BF16,
            bt: DataType::BF16,
            dt: DataType::BF16,
            gemm_tiling: tiling,
            transpose_b: true,
            b_prologue: GemmBPrologueKind::FullPrecision,
            bits: 0,
            group_size: 0,
        }
    }

    #[proc_macros::uzu_test]
    fn resolves_compiled_variant_without_runtime_mangling() {
        let request = AttentionGemmRequest {
            t: DataType::F16,
            bk: 32,
            bd: 128,
            use_mxu: true,
        };
        assert_eq!(request.resolve().unwrap(), "_D13AttentionGemmS4VhalfS2V32S3V128S4Vtrue");
    }

    #[proc_macros::uzu_test]
    fn rejects_variant_excluded_by_shader_constraints() {
        let request = AttentionGemmRequest {
            t: DataType::F32,
            bk: 16,
            bd: 64,
            use_mxu: true,
        };
        assert!(matches!(
            request.resolve(),
            Err(MetalError::UnsupportedKernelVariant {
                kernel: "AttentionGemm",
                request,
            }) if request.contains("t: F32") && request.contains("use_mxu: true")
        ));
    }

    #[proc_macros::uzu_test]
    fn gemm_requests_resolve_static_entries_for_simdgroup_and_derived_mxu_tilings() {
        assert_eq!(GemmRequest::ACCEPTED_VARIANT_COUNT, 676);
        assert_eq!(GemvRequest::ACCEPTED_VARIANT_COUNT, 800);

        let simdgroup = full_precision_gemm(GemmTiling::Tile64x64x32_Simdgroups2x2).resolve().unwrap();
        let mxu = full_precision_gemm(GemmTiling::Tile64x64x256_Simdgroups2x2).resolve().unwrap();
        assert_ne!(simdgroup, mxu);
    }

    #[proc_macros::uzu_test]
    fn gemv_request_resolves_compiled_static_entry() {
        let mut request = GemvRequest {
            at: DataType::BF16,
            bt: DataType::BF16,
            dt: DataType::BF16,
            b_prologue: GemmBPrologueKind::FullPrecision,
            group_size: 0,
            bits: 0,
            k_split: 1,
            input_aligned: true,
            results_per_simdgroup: 1,
            num_simdgroups: 8,
        };
        request.b_prologue = GemmBPrologueKind::ScaleSymmetricDequant;
        request.group_size = 32;
        request.bits = 4;
        assert_eq!(
            request.resolve().unwrap(),
            "_D4GemvS6VbfloatS6VbfloatS6VbfloatS21VScaleSymmetricDequantS2V32S1V4S1V1S4VtrueS1V1S1V8"
        );
    }

    #[proc_macros::uzu_test]
    fn invalid_grouped_variant_returns_contextual_error() {
        let mut request = GemvRequest {
            at: DataType::BF16,
            bt: DataType::BF16,
            dt: DataType::BF16,
            b_prologue: GemmBPrologueKind::FullPrecision,
            group_size: 0,
            bits: 4,
            k_split: 1,
            input_aligned: true,
            results_per_simdgroup: 1,
            num_simdgroups: 8,
        };

        assert!(matches!(
            request.resolve(),
            Err(MetalError::UnsupportedKernelVariant {
                kernel: "Gemv",
                request,
            }) if request.contains("b_prologue: FullPrecision") && request.contains("bits: 4")
        ));
        request.b_prologue = GemmBPrologueKind::ScaleSymmetricDequant;
        assert!(matches!(request.resolve(), Err(MetalError::UnsupportedKernelVariant { .. })));
    }
}
