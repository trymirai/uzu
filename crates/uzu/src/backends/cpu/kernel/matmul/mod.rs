use crate::backends::{
    common::kernel::matmul::{MatmulKernel, MatmulKernels},
    cpu::{Cpu, kernel::CpuKernels},
};

pub mod gemm;
pub mod gemv;

impl MatmulKernels for CpuKernels {
    type FullPrecisionMatmulKernel = MatmulKernel<Cpu>;
}
