use crate::backends::common::gpu_types::gemm::GemmDTransform;

#[derive(Debug, Clone, Copy)]
pub struct MatmulShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub b_transpose: bool,
    pub b_leading_dimension: Option<u32>,
    pub is_quant: bool,
    pub b_bits: Option<u32>,
    pub b_group_size: Option<u32>,
    pub gathered: bool,
    pub d_transform: GemmDTransform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulPath {
    Gemv,
    Gemm,
}
