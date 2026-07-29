use super::MatmulArguments;
use crate::backends::common::{Backend, BufferArg, gpu_types::gemm::GemmDTransform};

#[derive(Debug, Clone, Copy)]
pub struct MatmulShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub b_transpose: bool,
    pub b_leading_dimension: Option<u32>,
    pub b_bits: Option<u32>,
    pub b_group_size: Option<u32>,
    pub gathered: bool,
    pub d_transform: GemmDTransform,
}

impl MatmulShape {
    pub fn from_arguments<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
        arguments: &MatmulArguments<'a, 'b, 'd, B, TB>
    ) -> Self {
        Self {
            m: arguments.m,
            n: arguments.n,
            k: arguments.k,
            b_transpose: arguments.b_transpose,
            b_leading_dimension: arguments.b_leading_dimension,
            b_bits: arguments.b.bits_per_b(),
            b_group_size: arguments.b.group_size(),
            gathered: arguments.gather_indices.is_some(),
            d_transform: arguments.d_transform.mask(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulPath {
    Gemv,
    Gemm,
}
