//! GEMM kernel parameter structs.
//!
//! Field names use uppercase for matrix dimensions (M, N, K) to match
//! standard BLAS/GEMM convention and generated shader headers.

#![allow(non_snake_case)]

/// Main GEMM parameters passed to backend kernels as constant buffer.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GEMMParams {
    pub M: i32,
    pub N: i32,
    pub K: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldd: i32,
    pub tiles_n: i32,
    pub tiles_m: i32,
    pub swizzle_log: i32,
    pub gemm_k_iterations_aligned: i32,
}
