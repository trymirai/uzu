use super::{d_ops::MatmulDOps, matmul_a::MatmulA, matmul_b::MatmulB};
use crate::backends::common::{Allocation, Backend, BufferArg};

pub struct MatmulArguments<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B> = &'b Allocation<B>> {
    pub a: MatmulA<'a, B>,
    pub b: MatmulB<'b, B, TB>,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub d: &'d mut Allocation<B>,
    pub d_transform: MatmulDOps<'d, B>,
    pub gather_indices: Option<&'a Allocation<B>>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
