pub mod gemm;
pub mod gemv;
mod matmul;
pub(crate) mod quant;

pub use matmul::MatmulCpuKernel;

use crate::backends::{common::kernel::ManualKernels, cpu::kernel::CpuKernels};

impl ManualKernels for CpuKernels {
    type MatmulKernel = MatmulCpuKernel;
}
