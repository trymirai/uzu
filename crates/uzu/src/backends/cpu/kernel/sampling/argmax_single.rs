use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, ArgmaxSingleKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct ArgmaxSingleCpuKernel;

impl ArgmaxSingleKernel for ArgmaxSingleCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits_data, 'final_tokens, 'encoder>(
        &self,
        _logits_data: impl BufferArg<'logits_data, <Self::Backend as Backend>::NativeBuffer>,
        _final_tokens: impl BufferArg<'final_tokens, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits_data, 'final_tokens, 'encoder, 'predicate>(
        &self,
        _logits_data: impl BufferArg<'logits_data, <Self::Backend as Backend>::NativeBuffer>,
        _final_tokens: impl BufferArg<'final_tokens, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
