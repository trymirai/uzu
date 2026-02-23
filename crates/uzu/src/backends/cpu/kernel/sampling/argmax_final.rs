use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, ArgmaxFinalKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct ArgmaxFinalCpuKernel;

impl ArgmaxFinalKernel for ArgmaxFinalCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'partial_results, 'final_tokens, 'encoder>(
        &self,
        _partial_results: impl BufferArg<'partial_results, <Self::Backend as Backend>::NativeBuffer>,
        _final_tokens: impl BufferArg<'final_tokens, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'partial_results, 'final_tokens, 'encoder, 'predicate>(
        &self,
        _partial_results: impl BufferArg<'partial_results, <Self::Backend as Backend>::NativeBuffer>,
        _final_tokens: impl BufferArg<'final_tokens, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
