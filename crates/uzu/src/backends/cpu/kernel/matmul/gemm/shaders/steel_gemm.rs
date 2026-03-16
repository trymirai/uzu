use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    ArrayElement,
    pointers::{SendPtr, SendPtrMut},
};

#[inline(always)]
unsafe fn matmul_gemm_compute_row<T: ArrayElement + Float>(
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

fn matmul_gemm_impl<T: ArrayElement + Float>(
    a: *const T,
    b: *const T,
    d: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldd: usize,
) {
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let a = SendPtr(a);
    let b = SendPtr(b);
    let d = SendPtrMut(d);

    unsafe {
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

                s.spawn(move || unsafe {
                    for row in start..end {
                        matmul_gemm_compute_row(a, b, d, row, n, k, lda, ldb, ldd);
                    }
                });

                start = end;
            }
        });
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
    unsafe {
        let p = &*params.as_ptr();
        matmul_gemm_impl::<T>(
            a,
            b,
            d,
            p.M as usize,
            p.N as usize,
            p.K as usize,
            p.lda as usize,
            p.ldb as usize,
            p.ldd as usize,
        );
    }
}
