pub mod gemm;
pub mod gemm_mpp;
pub mod gemm_mpp_direct;
pub mod gemv;
mod matmul;

pub use matmul::MatmulMetalKernel;
