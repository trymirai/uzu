mod error;
pub mod gemm;
mod gemv;
mod matmul_dispatch;
mod matmul_metal_kernel;
mod matmul_output_work;

pub use error::MatmulMetalError;
pub use gemm::GemmKernel;
#[cfg(test)]
use matmul_dispatch::MatmulDispatch;
pub use matmul_metal_kernel::MatmulMetalKernel;
use matmul_metal_kernel::supports_integer_right_operand;
pub use matmul_output_work::MatmulOutputWork;

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/matmul/mod.rs"]
mod tests;
