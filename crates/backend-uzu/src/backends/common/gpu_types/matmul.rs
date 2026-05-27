#![allow(non_snake_case)]

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GemmParams {
    pub M: u32,
    pub N: u32,
    pub K: u32,
    pub leading_dimension_a: u32,
    pub leading_dimension_b: u32,
    pub leading_dimension_d: u32,
    pub threadgroups_per_column: u32,
    pub threadgroups_per_row: u32,
    pub aligned_inner_iterations: u32,
    pub use_morton: bool,
    pub ab_scale: f32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GemvParams {
    pub in_vec_size: u32,
    pub out_vec_size: u32,
    pub batch_size: u32,
    pub matrix_leading_dimension: u32,
    pub output_rows_per_threadgroup: u32,
    pub ab_scale: f32,
}
