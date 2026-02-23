use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, TokenCopyToResultsKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct TokenCopyToResultsCpuKernel;

impl TokenCopyToResultsKernel for TokenCopyToResultsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'src, 'dst, 'encoder>(
        &self,
        _src: impl BufferArg<'src, <Self::Backend as Backend>::NativeBuffer>,
        _dst: impl BufferArg<'dst, <Self::Backend as Backend>::NativeBuffer>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'src, 'dst, 'encoder, 'predicate>(
        &self,
        _src: impl BufferArg<'src, <Self::Backend as Backend>::NativeBuffer>,
        _dst: impl BufferArg<'dst, <Self::Backend as Backend>::NativeBuffer>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
