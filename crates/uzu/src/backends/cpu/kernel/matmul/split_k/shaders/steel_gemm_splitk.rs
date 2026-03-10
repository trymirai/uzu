use dsl::kernel;
use half::bf16;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MatmulSplitKPartialBfloat16)]
pub fn matmul_split_k_partial_bfloat16(
    #[allow(unused)] a: *const bf16,
    #[allow(unused)] b: *const bf16,
    #[allow(unused)] c: *mut f32,
    #[allow(unused)] params: &[crate::backends::common::gpu_types::matmul::GEMMSpiltKParams],
    #[allow(unused)] partial_group_count_x: u32,
    #[allow(unused)] partial_group_count_y: u32,
    #[allow(unused)] partial_group_count_z: u32,
) {
    todo!()
}

#[kernel(MatmulSplitKAccumBfloat16)]
pub fn matmul_split_k_accum_bfloat16(
    #[allow(unused)] c_split: *const f32,
    #[allow(unused)] d: *mut bf16,
    #[allow(unused)] k_partitions: i32,
    #[allow(unused)] partition_stride: i32,
    #[allow(unused)] ldd: i32,
    #[allow(unused)] accum_total_threads_x: u32,
    #[allow(unused)] accum_total_threads_y: u32,
) {
    todo!()
}
