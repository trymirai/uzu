use crate::backends::common::Backend;

pub mod attention_gemm;
pub mod delta_net_chunked_prefill;
pub mod delta_net_tree_verify;
pub mod matmul;
pub mod radix_top_k_small;
pub(crate) mod weaver;

include!(concat!(env!("OUT_DIR"), "/traits.rs"));

pub trait Kernels: Sized {
    type Backend: Backend<Kernels = Self>;

    autogen_kernels!();
    type AttentionGemmCore: attention_gemm::AttentionGemmCore<Self::Backend>;
    type DeltaNetChunkedPrefill: delta_net_chunked_prefill::DeltaNetChunkedPrefill<Self::Backend>;
    type DeltaNetTreeVerify: delta_net_tree_verify::DeltaNetTreeVerify<Self::Backend>;
    type MatmulKernel: matmul::MatmulKernel<Backend = Self::Backend>;
    type RadixTopKSmall: radix_top_k_small::RadixTopKSmall<Self::Backend>;
}

#[cfg(test)]
#[path = "../../../../unit/backends/common/kernel/mod.rs"]
mod tests;
