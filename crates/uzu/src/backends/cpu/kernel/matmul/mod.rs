pub mod gemm;
pub mod gemv;
mod matmul;

pub use matmul::MatmulCpuKernel;

use crate::backends::{common::kernel::matmul::MatmulKernels, cpu::kernel::CpuKernels};

impl MatmulKernels for CpuKernels {
    type MatmulKernel = MatmulCpuKernel;
}
