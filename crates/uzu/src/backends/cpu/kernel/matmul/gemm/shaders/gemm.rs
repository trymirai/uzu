use dsl::kernel;
use half::{bf16, f16};

#[kernel(MatmulGemm)]
#[variants(T, f16, bf16)]
pub fn matmul_gemm<T>(
    #[allow(unused)] a: *const T,
    #[allow(unused)] b: *const T,
    #[allow(unused)] d: *mut T,
    #[allow(unused)] params: &[crate::backends::common::gpu_types::matmul::GEMMParams],
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
    todo!()
}
