use super::{d_ops::MatmulDOps, matmul_b::MatmulB, task::MatmulTask};
use crate::backends::common::{Allocation, AsBufferRangeRef, Backend};

pub struct MatmulArguments<'a, B: Backend, TB: AsBufferRangeRef = Allocation<B>> {
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    pub b: MatmulB<'a, B, TB>,
    pub b_offset: usize,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub d: &'a mut Allocation<B>,
    pub d_transform: MatmulDOps<'a, B>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl<B: Backend, TB: AsBufferRangeRef> MatmulArguments<'_, B, TB> {
    pub fn task(&self) -> MatmulTask {
        MatmulTask {
            m: self.m,
            n: self.n,
            k: self.k,
            b_transpose: self.b_transpose,
            b_offset: self.b_offset,
            b_leading_dimension: self.b_leading_dimension,
            b_prologue: self.b.b_prologue(),
            bits: self.b.bits_per_b(),
            group_size: self.b.group_size(),
            d_transform: self.d_transform.mask(),
        }
    }
}
