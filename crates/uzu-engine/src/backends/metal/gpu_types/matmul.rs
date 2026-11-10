pub const MXU_TILED_BLOCK_ROWS: u32 = 32;
pub const MXU_TILED_W2_BLOCK_ROWS: u32 = 16;
pub const MXU_TILED_PREPARE_THREADS: u32 = 1024;
pub const INTERLEAVED_ROWS_STRIP_ROWS: u32 = 32;
pub const INTERLEAVED_ROWS_LANE_ROWS: u32 = 8;
pub const INTERLEAVED_ROWS_RELAYOUT_THREADS: u32 = 1024;

#[repr(C)]
#[allow(non_snake_case)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GemmParams {
    pub M: u32,
    pub N: u32,
    pub K: u32,
    pub leading_dimension_a: u32,
    pub leading_dimension_b: u32,
    pub scale_output_stride: u32,
    pub scale_group_stride: u32,
    pub zero_point_output_stride: u32,
    pub zero_point_group_stride: u32,
    pub leading_dimension_d: u32,
    pub threadgroups_per_column: u32,
    pub threadgroups_per_row: u32,
    pub aligned_inner_iterations: u32,
    pub use_morton: bool,
    pub ab_scale: f32,
}
