use super::{MatmulOutput, matmul_a::MatmulA, matmul_b::MatmulB};
use crate::backends::common::{Backend, BufferMut, BufferRef};

pub struct MatmulArguments<
    'd,
    B: Backend,
    TA: BufferRef<Backend = B>,
    TB: BufferRef<Backend = B>,
    TD: BufferMut<Backend = B>,
    TI: BufferRef<Backend = B>,
> {
    pub a: MatmulA<TA>,
    pub b: MatmulB<TB>,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub output: MatmulOutput<'d, B, TD>,
    pub gather_indices: Option<TI>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
