use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    ArrayElement,
    pointers::{SendPtr, SendPtrMut},
};

#[inline(always)]
unsafe fn compute_gemv_rows<T: ArrayElement + Float>(
    mat: SendPtr<T>,
    vec_row: SendPtr<T>,
    out_row: SendPtrMut<T>,
    src: Option<SendPtr<T>>,
    out_start: usize,
    out_end: usize,
    in_dim: usize,
    mat_ld: usize,
    output_scale: f32,
    output_accumulate_scale: f32,
    output_source_stride: usize,
    apply_output_scale_and_accumulate: bool,
) {
    unsafe {
        for out_idx in out_start..out_end {
            let mat_row = mat.0.add(out_idx * mat_ld);

            let mut acc: f32 = 0.0;
            for in_idx in 0..in_dim {
                acc += (*mat_row.add(in_idx)).to_f32().unwrap() * (*vec_row.0.add(in_idx)).to_f32().unwrap();
            }

            if apply_output_scale_and_accumulate {
                if let Some(s) = src {
                    let src_val = (*s.0.add(out_idx * output_source_stride)).to_f32().unwrap();
                    acc = output_scale * acc + output_accumulate_scale * src_val;
                }
            }

            *out_row.0.add(out_idx) = T::from(acc).unwrap();
        }
    }
}

fn matmul_gemv_multi_thread<T: ArrayElement + Float>(
    matrix: *const T,
    input_vector: *const T,
    output_source: Option<*const T>,
    output_vector: *mut T,
    input_dimension: usize,
    output_dimension: usize,
    matrix_leading_dimension: usize,
    output_scale: f32,
    output_accumulate_scale: f32,
    batches_count: usize,
    vector_batch_stride: &[i32],
    matrix_batch_stride: &[i32],
    output_source_batch_stride: &[i32],
    output_source_stride: usize,
    batch_rows: usize,
    apply_output_scale_and_accumulate: bool,
) {
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    unsafe {
        for outer_batch in 0..batches_count as usize {
            let batch_vec = input_vector.add(outer_batch * vector_batch_stride[0] as usize);
            let batch_mat = SendPtr(matrix.add(outer_batch * matrix_batch_stride[0] as usize));
            let batch_out = SendPtrMut(output_vector.add(outer_batch * batch_rows * output_dimension));

            let batch_src =
                output_source.map(|src| SendPtr(src.add(outer_batch * output_source_batch_stride[0] as usize)));

            for batch_row in 0..batch_rows {
                let vec_row = SendPtr(batch_vec.add(batch_row * input_dimension));
                let out_row = SendPtrMut(batch_out.0.add(batch_row * output_dimension));

                std::thread::scope(|s| {
                    let rows_per_thread = output_dimension / num_threads;
                    let rows_per_thread_remainder = output_dimension % num_threads;
                    let mut start = 0;

                    for t in 0..num_threads {
                        let chunk = rows_per_thread
                            + if t < rows_per_thread_remainder {
                                1
                            } else {
                                0
                            };
                        let end = start + chunk;

                        s.spawn(move || unsafe {
                            compute_gemv_rows(
                                batch_mat,
                                vec_row,
                                out_row,
                                batch_src,
                                start,
                                end,
                                input_dimension,
                                matrix_leading_dimension,
                                output_scale,
                                output_accumulate_scale,
                                output_source_stride,
                                apply_output_scale_and_accumulate,
                            );
                        });

                        start = end;
                    }
                });
            }
        }
    }
}

fn matmul_gemv_single_thread<T: ArrayElement + Float>(
    matrix: *const T,
    input_vector: *const T,
    output_source: Option<*const T>,
    output_vector: *mut T,
    input_dimension: usize,
    output_dimension: usize,
    matrix_leading_dimension: usize,
    output_scale: f32,
    output_accumulate_scale: f32,
    batches_count: usize,
    vector_batch_stride: &[i32],
    matrix_batch_stride: &[i32],
    output_source_batch_stride: &[i32],
    output_source_stride: usize,
    batch_rows: usize,
    apply_output_scale_and_accumulate: bool,
) {
    unsafe {
        for outer_batch in 0..batches_count {
            let batch_vec = input_vector.add(outer_batch * vector_batch_stride[0] as usize);
            let batch_mat = matrix.add(outer_batch * matrix_batch_stride[0] as usize);
            let batch_out = output_vector.add(outer_batch * batch_rows * output_dimension);

            let batch_src = output_source.map(|src| src.add(outer_batch * output_source_batch_stride[0] as usize));

            for batch_row in 0..batch_rows {
                let vec_row = batch_vec.add(batch_row * input_dimension);
                let out_row = batch_out.add(batch_row * output_dimension);

                for out_idx in 0..output_dimension {
                    let mat_row = batch_mat.add(out_idx * matrix_leading_dimension);

                    let mut acc: f32 = 0.0;
                    for in_idx in 0..input_dimension {
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
    matmul_gemv_multi_thread::<T>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension as usize,
        output_dimension as usize,
        matrix_leading_dimension as usize,
        output_scale,
        output_accumulate_scale,
        batch_shape[0] as usize,
        vector_batch_stride,
        matrix_batch_stride,
        output_source_batch_stride,
        output_source_stride as usize,
        batch_rows as usize,
        apply_output_scale_and_accumulate,
    );
}
