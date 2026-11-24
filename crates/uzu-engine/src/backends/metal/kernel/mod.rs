use crate::backends::{
    common::{
        Kernels, gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE, kernel::tree_verify_blocks::TreeVerifyBlocks,
    },
    metal::{
        Metal,
        buffer::MetalBufferExt,
        gpu_types::{
            gemm::{gemm_tiling_simdgroups_per_column, gemm_tiling_simdgroups_per_k, gemm_tiling_simdgroups_per_row},
            matmul::{INTERLEAVED_ROWS_RELAYOUT_THREADS, MXU_TILED_PREPARE_THREADS},
        },
    },
};

const METAL_SIMD_SIZE: u32 = 32;

const _: () = {
    assert!(HADAMARD_TRANSFORM_BLOCK_SIZE == METAL_SIMD_SIZE);
};

mod attention;
pub mod gdn;
mod indexer;
pub mod matmul;
mod radix_top_k_small;

include!(concat!(env!("OUT_DIR"), "/metal.rs"));

pub struct MetalKernels;

impl Kernels for MetalKernels {
    type Backend = Metal;

    autogen_kernels!();
    type AttentionKernel = attention::AttentionMetalKernel;
    type IndexerKernel = indexer::IndexerMetalKernel;
    type DeltaNetChunkedPrefill = gdn::chunked::MetalDeltaNetChunkedPrefill;
    type DeltaNetTreeVerify = TreeVerifyBlocks<Metal>;
    type MatmulKernel = matmul::MatmulMetalKernel;
    type RadixTopKSmall = radix_top_k_small::MetalRadixTopKSmall;
}
