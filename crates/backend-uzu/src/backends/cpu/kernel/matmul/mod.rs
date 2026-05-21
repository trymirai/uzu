pub mod gemm;
mod matmul;
mod quant;

pub use matmul::MatmulCpuKernel;
pub use quant::QuantizedGemmCpuKernel;

use crate::backends::{common::kernel::ManualKernels, cpu::kernel::CpuKernels};

impl ManualKernels for CpuKernels {
    type MatmulKernel = MatmulCpuKernel;
    type QuantizedGemmKernel = QuantizedGemmCpuKernel;
}
