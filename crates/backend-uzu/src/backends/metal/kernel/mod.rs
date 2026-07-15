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
}
