mod error;
pub mod gemm;
mod gemv;
mod layout;
mod matmul_dispatch;
mod matmul_metal_kernel;
mod matmul_output_work;

const U2_BLOCK_K: u32 = 128;

fn weight_block_k(
    bits: Option<u32>,
    group_size: Option<u32>,
) -> Option<u32> {
    group_size.map(|group_size| {
        if bits == Some(2) {
            U2_BLOCK_K
        } else {
            group_size
        }
    })
}

pub use error::MatmulMetalError;
pub use gemm::GemmKernel;
pub use matmul_metal_kernel::MatmulMetalKernel;
use matmul_metal_kernel::supports_integer_right_operand;
pub use matmul_output_work::MatmulOutputWork;

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/matmul/mod.rs"]
mod tests;
