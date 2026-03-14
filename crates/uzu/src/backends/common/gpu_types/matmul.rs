//! GEMM kernel parameter structs.
//!
//! Field names use uppercase for matrix dimensions (M, N, K) to match
//! standard BLAS/GEMM convention and generated shader headers.

#![allow(non_snake_case)]

use crate::backends::common::GridSize;

/// Main GEMM parameters passed to backend kernels as constant buffer.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GEMMParams {
    pub M: i32,
    pub N: i32,
    pub K: i32,
    pub leading_dim_a: i32,
    pub leading_dim_b: i32,
    pub leading_dim_d: i32,
    pub tiles_n: i32,
    pub tiles_m: i32,
    pub batch_stride_a: i64,
    pub batch_stride_b: i64,
    pub batch_stride_d: i64,
    pub swizzle_log: i32,
    pub gemm_k_iterations_aligned: i32,
}

impl GEMMParams {
    pub fn with_grid(
        m: i32,
        n: i32,
        k: i32,
        block_rows: i32,
        block_cols: i32,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
    ) -> (Self, GridSize) {
        let tiles_n = (n + block_cols - 1) / block_cols;
        let tiles_m = (m + block_rows - 1) / block_rows;

        let tile_swizzle = 1 << swizzle_log;
        let tm_swizzled = (tiles_m + tile_swizzle - 1) / tile_swizzle;
        let tn_swizzled = tiles_n * tile_swizzle;

        let params = Self {
            M: m,
            N: n,
            K: k,
            leading_dim_a: k,
            leading_dim_b: k,
            leading_dim_d: n,
            tiles_n,
            tiles_m,
            batch_stride_a: (m as i64) * (k as i64),
            batch_stride_b: (n as i64) * (k as i64),
            batch_stride_d: (m as i64) * (n as i64),
            swizzle_log,
            gemm_k_iterations_aligned,
        };

        let grid = GridSize {
            width: tn_swizzled as usize,
            height: tm_swizzled as usize,
            depth: 1,
        };

        (params, grid)
    }
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
