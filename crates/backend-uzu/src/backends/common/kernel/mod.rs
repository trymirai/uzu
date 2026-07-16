use crate::backends::common::Backend;

pub mod activation_prepare;
pub mod attention_gemm;
pub mod delta_net_chunked_prefill;
pub mod delta_net_tree_verify;
pub mod matmul;

pub use activation_prepare::{
    ActivationPrepareConfig, INT8_SYMMETRIC_QMAX, group_stat, quantize_symmetric_i8, symmetric_divisor,
};

include!(concat!(env!("OUT_DIR"), "/traits.rs"));

pub trait Kernels: Sized {
    type Backend: Backend<Kernels = Self>;

    autogen_kernels!();
    type AttentionGemmCore: attention_gemm::AttentionGemmCore<Self::Backend>;
    type DeltaNetChunkedPrefill: delta_net_chunked_prefill::DeltaNetChunkedPrefill<Self::Backend>;
    type DeltaNetTreeVerify: delta_net_tree_verify::DeltaNetTreeVerify<Self::Backend>;
    type MatmulKernel: matmul::MatmulKernel<Backend = Self::Backend>;
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/common/kernel/mod.rs"]
mod tests;
