pub mod matmul;
pub mod unified_matmul;

use crate::backends::common::kernel::ManualKernels;

include!(concat!(env!("OUT_DIR"), "/dsl.rs"));

pub(super) const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

impl ManualKernels for MetalKernels {
    type MatmulKernel = matmul::MatmulMetalKernel;
}
