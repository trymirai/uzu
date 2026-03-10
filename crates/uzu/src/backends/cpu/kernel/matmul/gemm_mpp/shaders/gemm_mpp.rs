use dsl::kernel;
use half::{bf16, f16};

#[kernel(MatmulGemmMpp)]
#[variants(AType, f32, f16, bf16, i8)]
#[variants(BType, f32, f16, bf16, i8)]
#[variants(OutType, f32, f16, bf16, i32)]
pub fn matmul_gemm_mpp<AType, BType, OutType>(
    #[allow(unused)] a: *const AType,
    #[allow(unused)] b: *const BType,
    #[allow(unused)] d: *mut OutType,
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
    #[allow(unused)]
    #[specialize]
    use_native_fragment_layout: bool,
) {
    todo!()
}
