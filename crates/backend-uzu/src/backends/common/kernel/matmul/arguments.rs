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
        MatmulTask::new(
            self.m,
            self.n,
            self.k,
            self.b_transpose,
            self.b_offset,
            self.b_leading_dimension,
            &self.b,
            &self.d_transform,
        )
    }
}
