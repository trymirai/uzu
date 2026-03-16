#![allow(non_snake_case)]

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GemmParams {
    pub M: i32,
    pub N: i32,
    pub K: i32,
    pub leading_dimension_a: i32,
    pub leading_dimension_b: i32,
    pub leading_dimension_d: i32,
    pub threadgroups_per_row: i32,
    pub threadgroups_per_column: i32,
    pub swizzle_log: i32,
    pub aligned_inner_iterations: i32,
}
