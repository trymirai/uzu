use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MatmulGemv)]
#[variants(T, f32, f16, bf16)]
pub fn matmul_gemv<T: ArrayElement + Float>(
    #[allow(unused)] matrix: *const T,
    #[allow(unused)] input_vector: *const T,
    #[allow(unused)]
    #[optional(apply_output_scale_and_accumulate)]
    output_source: Option<*const T>,
    #[allow(unused)] output_vector: *mut T,
    #[allow(unused)] input_dimension: i32,
    #[allow(unused)] output_dimension: i32,
    #[allow(unused)] matrix_leading_dimension: i32,
    #[allow(unused)] output_scale: f32,
    #[allow(unused)] output_accumulate_scale: f32,
    #[allow(unused)] batch_shape: &[i32],
    #[allow(unused)] vector_batch_stride: &[i32],
    #[allow(unused)] matrix_batch_stride: &[i32],
    #[allow(unused)] output_source_batch_stride: &[i32],
    #[allow(unused)] output_source_stride: i32,
    #[allow(unused)] batch_rows: i32,
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
    #[allow(unused)]
    #[specialize]
    apply_output_scale_and_accumulate: bool,
) {
    todo!()
}
