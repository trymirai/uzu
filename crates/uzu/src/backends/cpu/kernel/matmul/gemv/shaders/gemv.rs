use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MatmulGemv)]
#[variants(T, f32, f16, bf16)]
pub fn matmul_gemv<T: ArrayElement + Float>(
    matrix: *const T,
    input_vector: *const T,
    #[optional(apply_output_scale_and_accumulate)] output_source: Option<*const T>,
    output_vector: *mut T,
    input_dimension: i32,
    output_dimension: i32,
    matrix_leading_dimension: i32,
    output_scale: f32,
    output_accumulate_scale: f32,
    batch_shape: &[i32],
    vector_batch_stride: &[i32],
    matrix_batch_stride: &[i32],
    output_source_batch_stride: &[i32],
    output_source_stride: i32,
    batch_rows: i32,
    #[allow(unused)] output_rows_per_threadgroup: i32,
    #[allow(unused)]
    #[specialize]
    tg_simd_rows: u32,
    #[allow(unused)]
    #[specialize]
    tg_simd_cols: u32,
    #[allow(unused)]
    #[specialize]
    sg_thread_rows: u32,
    #[allow(unused)]
    #[specialize]
    sg_thread_cols: u32,
    #[allow(unused)]
    #[specialize]
    thread_out_rows: u32,
    #[allow(unused)]
    #[specialize]
    thread_out_cols: u32,
    #[specialize] apply_output_scale_and_accumulate: bool,
) {
    let in_dim = input_dimension as usize;
    let out_dim = output_dimension as usize;
    let mat_ld = matrix_leading_dimension as usize;
    let num_batch_rows = batch_rows as usize;

    unsafe {
        for outer_batch in 0..batch_shape[0] as usize {
            let batch_vec = input_vector.add(outer_batch * vector_batch_stride[0] as usize);
            let batch_mat = matrix.add(outer_batch * matrix_batch_stride[0] as usize);
            let batch_out = output_vector.add(outer_batch * num_batch_rows * out_dim);

            let batch_src = output_source.map(|src| src.add(outer_batch * output_source_batch_stride[0] as usize));

            for batch_row in 0..num_batch_rows {
                let vec_row = batch_vec.add(batch_row * in_dim);
                let out_row = batch_out.add(batch_row * out_dim);

                for out_idx in 0..out_dim {
                    let mat_row = batch_mat.add(out_idx * mat_ld);

                    let mut acc: f32 = 0.0;
                    for in_idx in 0..in_dim {
                        acc += (*mat_row.add(in_idx)).to_f32().unwrap() * (*vec_row.add(in_idx)).to_f32().unwrap();
                    }

                    if apply_output_scale_and_accumulate {
                        if let Some(src) = batch_src {
                            let src_val = (*src.add(out_idx * output_source_stride as usize)).to_f32().unwrap();
                            acc = output_scale * acc + output_accumulate_scale * src_val;
                        }
                    }

                    *out_row.add(out_idx) = T::from(acc).unwrap();
                }
            }
        }
    }
}
