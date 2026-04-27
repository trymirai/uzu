pub mod gemm;
mod matmul;

pub use matmul::MatmulCpuKernel;

use crate::backends::{common::kernel::ManualKernels, cpu::kernel::CpuKernels};

impl ManualKernels for CpuKernels {
    type MatmulKernel = MatmulCpuKernel;
}
