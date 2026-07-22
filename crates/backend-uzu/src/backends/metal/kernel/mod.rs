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
