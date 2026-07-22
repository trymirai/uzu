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

/// A kernel variant that was never instantiated. Raised by the generated `validate()`
/// on each kernel's key, which looks the variant up in the set the build compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{kernel} was not instantiated for this key")]
pub struct InvalidKernelKey {
    pub kernel: &'static str,
}

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

/// The built set is written by the build script from each variant's declared text, and
/// `entry_name()` re-renders the same variant at runtime from the key's own values. The
/// two spellings have to agree, and nothing in the type system says they must: a
/// `Display` impl on an axis enum, or a data type spelled differently, would move one
/// side and not the other.
///
/// For GEMM and attention a disagreement is a loud `InvalidKernelKey`. For GEMV it is
/// silent -- `validate()` is the fallback predicate there, so every dispatch would just
/// quietly take the GEMM path. These three keys name variants the build always compiles,
/// so a disagreement fails here instead.
#[cfg(test)]
mod entry_names_agree_with_the_built_set {
    use super::{AttentionGemmKey, GemmKey, GemvKey};
    use crate::{
        backends::common::gpu_types::gemm::{GemmTiling, QuantBits, QuantGroupSize, QuantPrologue, WeightsKey},
        data_type::DataType,
    };

    const QUANTIZED: WeightsKey = WeightsKey::Quant {
        b_prologue: QuantPrologue::ScaleBiasDequant,
        bits: QuantBits::B4,
        group_size: QuantGroupSize::G64,
    };

    #[proc_macros::uzu_test]
    fn gemm() {
        let key = GemmKey {
            at: DataType::BF16,
            bt: DataType::BF16,
            dt: DataType::BF16,
            gemm_tiling: GemmTiling::Tile64x64x256_Simdgroups2x2,
            transpose_b: true,
            weights_key: QUANTIZED,
        };
        assert!(key.validate().is_ok(), "{} is not in GEMM_BUILT", key.entry_name());
    }

    #[proc_macros::uzu_test]
    fn gemv() {
        let key = GemvKey {
            at: DataType::BF16,
            bt: DataType::BF16,
            dt: DataType::BF16,
            weights_key: QUANTIZED,
            k_split: 1,
            input_aligned: true,
            results_per_simdgroup: 4,
            num_simdgroups: 8,
        };
        assert!(key.validate().is_ok(), "{} is not in GEMV_BUILT", key.entry_name());
    }

    #[proc_macros::uzu_test]
    fn attention_gemm() {
        let key = AttentionGemmKey {
            t: DataType::BF16,
            bk: 32,
            bd: 128,
            use_mxu: true,
        };
        assert!(key.validate().is_ok(), "{} is not in ATTENTION_GEMM_BUILT", key.entry_name());
    }

    /// And the other direction: a variant the build never compiles is rejected rather
    /// than dispatched to a pipeline that does not exist.
    #[proc_macros::uzu_test]
    fn an_unbuilt_variant_is_rejected() {
        let key = GemvKey {
            at: DataType::BF16,
            bt: DataType::BF16,
            dt: DataType::BF16,
            weights_key: QUANTIZED,
            k_split: 8,
            input_aligned: true,
            results_per_simdgroup: 4,
            num_simdgroups: 8,
        };
        assert!(key.validate().is_err(), "{} should not be in GEMV_BUILT", key.entry_name());
    }
}
