use crate::backends::{
    common::kernel::matmul::{MatmulKernel, MatmulKernels},
    cpu::{Cpu, kernel::CpuKernels},
};

pub mod gemm;
pub mod gemm_mpp;
pub mod gemv;
pub mod split_k;

impl MatmulKernels for CpuKernels {
    type FullPrecisionMatmulKernel = MatmulKernel<Cpu>;
}
