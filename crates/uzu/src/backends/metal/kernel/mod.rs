use crate::backends::{
    common::kernel::matmul::{MatmulKernel, MatmulKernels},
    metal::Metal,
};
pub mod dsl {
    include!(concat!(env!("OUT_DIR"), "/dsl.rs"));
}
pub(super) const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

pub mod matmul;

impl MatmulKernels for dsl::MetalKernels {
    type FullPrecisionMatmulKernel = MatmulKernel<Metal>;
}
