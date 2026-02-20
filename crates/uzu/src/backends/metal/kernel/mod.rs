use crate::backends::common::kernel::matmul::MatmulKernels;

pub mod dsl {
    include!(concat!(env!("OUT_DIR"), "/dsl.rs"));
}
pub(super) const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

pub mod matmul;

pub use matmul::{MatmulArguments, MatmulKernel};

impl MatmulKernels for dsl::MetalKernels {
    type FullPrecisionMatmulKernel = MatmulKernel;
}
