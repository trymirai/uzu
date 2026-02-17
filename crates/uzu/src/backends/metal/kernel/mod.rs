use crate::backends::common::kernel::matmul::MatmulKernels;
pub mod dsl {
    include!(concat!(env!("OUT_DIR"), "/dsl.rs"));
}
pub(super) const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

pub mod matmul;
pub mod moe;
pub mod quant_matmul;

pub use matmul::{MatmulArguments, MatmulKernel};
pub use moe::{
    MoeBlockBasesArguments, MoeExpertsArguments, MoeExpertsError, MoeExpertsTwoPassArguments,
    MoeExpertsTwoPassDecodeKernels, MoeExpertsTwoPassPrefillKernel, MoeGatherArguments, MoeGatherKernels,
    MoeRouterTopKArguments, MoeRouterTopKKernel, MoeScatterArguments, MoeScatterError, MoeScatterKernels,
    MoeScatterWithMapArguments,
};

impl MatmulKernels for dsl::MetalKernels {
    type FullPrecisionMatmulKernel = MatmulKernel;
    type QuantizedMatmulKernel = quant_matmul::QuantizedMatmulKernel;
}
