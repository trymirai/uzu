use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, GumbelKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct GumbelCpuKernel;

impl GumbelKernel for GumbelCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits, 'batch_seeds, 'processed_logits, 'encoder>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _batch_seeds: impl BufferArg<'batch_seeds, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits, 'batch_seeds, 'processed_logits, 'encoder, 'predicate>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _batch_seeds: impl BufferArg<'batch_seeds, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
