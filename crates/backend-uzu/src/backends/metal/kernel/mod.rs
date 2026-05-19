pub mod matmul;

use crate::backends::common::{
    gpu_types::gemm::{gemm_tiling_smg_m, gemm_tiling_smg_n},
    kernel::ManualKernels,
};

include!(concat!(env!("OUT_DIR"), "/dsl.rs"));

pub(super) const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

impl ManualKernels for MetalKernels {
    type MatmulKernel = matmul::MatmulMetalKernel;
}
