mod error;
pub mod gemm;
mod gemv;
mod matmul_metal_kernel;
mod matmul_output_work;
mod selection;

pub use error::MatmulMetalError;
pub use gemm::GemmKernel;
use gemm::supports_integer_right_operand;
pub use matmul_metal_kernel::MatmulMetalKernel;
pub use matmul_output_work::MatmulOutputWork;

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/matmul/mod.rs"]
mod tests;
