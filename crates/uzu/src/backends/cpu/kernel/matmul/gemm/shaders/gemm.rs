use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    ArrayElement,
    pointers::{SendPtr, SendPtrMut},
};

#[inline(always)]
unsafe fn matmul_gemm_multi_thread_compute_row<T: ArrayElement + Float>(
    a: SendPtr<T>,
    b: SendPtr<T>,
    d: SendPtrMut<T>,
    row: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldd: usize,
) {
    unsafe {
        for col in 0..n {
            let mut acc: f32 = 0.0;
            for i in 0..k {
                let a_val = *a.0.add(row * lda + i);
                let b_val = *b.0.add(col * ldb + i);
                acc = acc + a_val.to_f32().unwrap() * b_val.to_f32().unwrap();
            }
            *d.0.add(row * ldd + col) = T::from(acc).unwrap();
        }
    }
}

fn matmul_gemm_multi_thread<T: ArrayElement + Float>(
    a: *const T,
    b: *const T,
    d: *mut T,
    batches_count: usize,
    m: usize,
    n: usize,
    k: usize,
    batch_stride_a: usize,
    batch_stride_b: usize,
    batch_stride_d: usize,
    lda: usize,
    ldb: usize,
    ldd: usize,
) {
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    unsafe {
        for batch in 0..batches_count {
            let a_batch = SendPtr(a.add(batch * batch_stride_a));
            let b_batch = SendPtr(b.add(batch * batch_stride_b));
            let d_batch = SendPtrMut(d.add(batch * batch_stride_d));

            std::thread::scope(|s| {
                let rows_per_thread = m / num_threads;
                let rows_per_thread_remainder = m % num_threads;
                let mut start = 0;

                for t in 0..num_threads {
                    let chunk_add = if t < rows_per_thread_remainder {
                        1
                    } else {
                        0
                    };
                    let chunk = rows_per_thread + chunk_add;

                    let end = start + chunk;
                    let ab = a_batch;
                    let bb = b_batch;
                    let db = d_batch;
                    s.spawn(move || unsafe {
                        for row in start..end {
                            matmul_gemm_multi_thread_compute_row(ab, bb, db, row, n, k, lda, ldb, ldd);
                        }
                    });

                    start = end;
                }
            });
        }
    }
}

fn matmul_gemm_single_thread<T: ArrayElement + Float>(
    a: *const T,
    b: *const T,
    d: *mut T,
    batches_count: usize,
    m: usize,
    n: usize,
    k: usize,
    batch_stride_a: usize,
    batch_stride_b: usize,
    batch_stride_d: usize,
    lda: usize,
    ldb: usize,
    ldd: usize,
) {
    unsafe {
        for batch in 0..batches_count {
            let a_batch = a.add(batch * batch_stride_a);
            let b_batch = b.add(batch * batch_stride_b);
            let d_batch = d.add(batch * batch_stride_d);

            for row in 0..m {
                for col in 0..n {
                    let mut acc: f32 = 0.0;
                    for i in 0..k {
                        // A[row, i]: row-major with leading dimension lda
                        let a_val = *a_batch.add(row * lda + i);
                        // B[col, i]: B is transposed, so B is [N, K] row-major with leading dimension ldb
                        let b_val = *b_batch.add(col * ldb + i);
                        acc = acc + a_val.to_f32().unwrap() * b_val.to_f32().unwrap();
                    }
                    *d_batch.add(row * ldd + col) = T::from(acc).unwrap();
                }
            }
        }
    }
}

#[kernel(MatmulGemm)]
#[variants(T, f32, f16, bf16)]
pub fn matmul_gemm<T: ArrayElement + Float>(
    a: *const T,
    b: *const T,
    d: *mut T,
    params: &[crate::backends::common::gpu_types::matmul::GEMMParams],
    #[allow(unused)] group_count_x: u32,
    #[allow(unused)] group_count_y: u32,
    #[allow(unused)] group_count_z: u32,
    #[allow(unused)]
    #[specialize]
    block_rows: u32,
    #[allow(unused)]
    #[specialize]
    block_cols: u32,
    #[allow(unused)]
    #[specialize]
    block_depth: u32,
    #[allow(unused)]
    #[specialize]
    warps_per_row: u32,
    #[allow(unused)]
    #[specialize]
    warps_per_col: u32,
    #[allow(unused)]
    #[specialize]
    align_m: bool,
    #[allow(unused)]
    #[specialize]
    align_n: bool,
    #[allow(unused)]
    #[specialize]
    align_k: bool,
) {
    let batches_count = group_count_z as usize;
    unsafe {
        let p = &*params.as_ptr();
        matmul_gemm_multi_thread::<T>(
            a,
            b,
            d,
            group_count_z as usize,
            p.M as usize,
            p.N as usize,
            p.K as usize,
            p.batch_stride_a as usize,
            p.batch_stride_b as usize,
            p.batch_stride_d as usize,
            p.leading_dim_a as usize,
            p.leading_dim_b as usize,
            p.leading_dim_d as usize,
        );
    }
}
