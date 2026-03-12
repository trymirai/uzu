use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

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
    unsafe {
        let p = &*params.as_ptr();
        let lda = p.lda as usize;
        let ldb = p.ldb as usize;
        let ldd = p.ldd as usize;

        for batch in 0..group_count_z as usize {
            let a_batch = a.add(batch * (p.batch_stride_a as usize));
            let b_batch = b.add(batch * (p.batch_stride_b as usize));
            let d_batch = d.add(batch * (p.batch_stride_d as usize));

            for row in 0..p.M as usize {
                for col in 0..p.N as usize {
                    let mut acc: f32 = 0.0;
                    for i in 0..p.K as usize {
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
