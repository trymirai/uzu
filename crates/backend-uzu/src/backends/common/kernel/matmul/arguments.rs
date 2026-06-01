use super::{d_ops::MatmulDOps, matmul_b::MatmulB};
use crate::backends::common::{Allocation, Backend, kernel::BufferArg};

pub struct MatmulArguments<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B> = &'b Allocation<B>> {
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    pub b: MatmulB<'b, B, TB>,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub d: &'d mut Allocation<B>,
    pub d_transform: MatmulDOps<'d, B>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
