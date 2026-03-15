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

fn matmul_gemv_impl<T: ArrayElement + Float>(
    matrix: *const T,
    input_vector: *const T,
    output_source: Option<*const T>,
    output_vector: *mut T,
    input_dimension: usize,
    output_dimension: usize,
    matrix_leading_dimension: usize,
    output_scale: f32,
    output_accumulate_scale: f32,
    output_source_stride: usize,
    batch_rows: usize,
    apply_output_scale_and_accumulate: bool,
) {
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let matrix = SendPtr(matrix);
    let output_source = output_source.map(SendPtr);

    unsafe {
        for batch_row in 0..batch_rows {
            let vec_row = SendPtr(input_vector.add(batch_row * input_dimension));
            let out_row = SendPtrMut(output_vector.add(batch_row * output_dimension));

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
                            matrix,
                            vec_row,
                            out_row,
                            output_source,
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
    matmul_gemv_impl::<T>(
        matrix,
        input_vector,
        output_source,
        output_vector,
        input_dimension as usize,
        output_dimension as usize,
        matrix_leading_dimension as usize,
        output_scale,
        output_accumulate_scale,
        output_source_stride as usize,
        batch_rows as usize,
        apply_output_scale_and_accumulate,
    );
}
