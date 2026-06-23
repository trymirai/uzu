use crate::backends::common::Backend;

pub mod attention_gemm;
pub mod matmul;

include!(concat!(env!("OUT_DIR"), "/traits.rs"));

pub trait Kernels: Sized {
    type Backend: Backend<Kernels = Self>;

    autogen_kernels!();
    type AttentionGemmBlock: attention_gemm::AttentionGemmBackendBlock<Backend = Self::Backend>;
    type MatmulKernel: matmul::MatmulKernel<Backend = Self::Backend>;
}

#[cfg(test)]
#[path = "../../../../unit/backends/common/kernel/mod.rs"]
mod tests;
