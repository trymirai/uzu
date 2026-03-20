pub mod gemm;
pub mod gemm_mpp;
pub mod gemm_mpp_mxu;
pub mod gemv;
mod matmul;

pub use matmul::MatmulMetalKernel;
