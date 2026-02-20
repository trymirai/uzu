//! GEMM kernel parameter structs.
//!
//! Field names use uppercase for matrix dimensions (M, N, K) to match
//! standard BLAS/GEMM convention and generated Metal headers.

#![allow(non_snake_case)]

/// Main GEMM parameters passed to Metal kernels as constant buffer.
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
    pub batch_stride_a: i64,
    pub batch_stride_b: i64,
    pub batch_stride_d: i64,
    pub swizzle_log: i32,
    pub gemm_k_iterations_aligned: i32,
    pub batch_ndim: i32,
}

/// Split-K GEMM parameters.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GEMMSpiltKParams {
    pub M: i32,
    pub N: i32,
    pub K: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub tiles_n: i32,
    pub tiles_m: i32,
    pub split_k_partitions: i32,
    pub split_k_partition_stride: i32,
    pub split_k_partition_size: i32,
    pub gemm_k_iterations_aligned: i32,
}

