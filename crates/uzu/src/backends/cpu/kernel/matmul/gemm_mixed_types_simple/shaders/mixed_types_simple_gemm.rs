use dsl::kernel;
use half::{bf16, f16};

#[kernel(MixedTypesSimpleGemmI8I8I32)]
pub fn mixed_types_simple_gemm_i8_i8_i32(
    #[allow(unused)] a: *const i8,
    #[allow(unused)] b: *const i8,
    #[allow(unused)] d: *mut i32,
    #[allow(unused)] params: &[crate::backends::common::gpu_types::matmul::GEMMParams],
    #[allow(unused)] group_count_x: u32,
    #[allow(unused)] group_count_y: u32,
    #[allow(unused)] group_count_z: u32,
) {
    todo!()
}

#[kernel(MixedTypesSimpleGemmI8Bf16Bf16)]
pub fn mixed_types_simple_gemm_i8_bf16_bf16(
    #[allow(unused)] a: *const i8,
    #[allow(unused)] b: *const bf16,
    #[allow(unused)] d: *mut bf16,
    #[allow(unused)] params: &[crate::backends::common::gpu_types::matmul::GEMMParams],
    #[allow(unused)] group_count_x: u32,
    #[allow(unused)] group_count_y: u32,
    #[allow(unused)] group_count_z: u32,
) {
    todo!()
}

#[kernel(MixedTypesSimpleGemmI8F16F16)]
pub fn mixed_types_simple_gemm_i8_f16_f16(
    #[allow(unused)] a: *const i8,
    #[allow(unused)] b: *const f16,
    #[allow(unused)] d: *mut f16,
    #[allow(unused)] params: &[crate::backends::common::gpu_types::matmul::GEMMParams],
    #[allow(unused)] group_count_x: u32,
    #[allow(unused)] group_count_y: u32,
    #[allow(unused)] group_count_z: u32,
) {
    todo!()
}

#[kernel(MixedTypesSimpleGemmI8F32F32)]
pub fn mixed_types_simple_gemm_i8_f32_f32(
    #[allow(unused)] a: *const i8,
    #[allow(unused)] b: *const f32,
    #[allow(unused)] d: *mut f32,
    #[allow(unused)] params: &[crate::backends::common::gpu_types::matmul::GEMMParams],
    #[allow(unused)] group_count_x: u32,
    #[allow(unused)] group_count_y: u32,
    #[allow(unused)] group_count_z: u32,
) {
    todo!()
}
