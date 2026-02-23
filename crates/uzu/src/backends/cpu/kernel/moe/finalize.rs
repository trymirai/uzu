use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeFinalizeKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeFinalizeCpuKernel;

impl MoeFinalizeKernel for MoeFinalizeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'tok2row, 'probs, 'y_partial, 'y, 'encoder>(
        &self,
        _tok2row: impl BufferArg<'tok2row, <Self::Backend as Backend>::NativeBuffer>,
        _probs: impl BufferArg<'probs, <Self::Backend as Backend>::NativeBuffer>,
        _y_partial: impl BufferArg<'y_partial, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _t_count: u32,
        _d_model: u32,
        _k_input: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'tok2row, 'probs, 'y_partial, 'y, 'encoder, 'predicate>(
        &self,
        _tok2row: impl BufferArg<'tok2row, <Self::Backend as Backend>::NativeBuffer>,
        _probs: impl BufferArg<'probs, <Self::Backend as Backend>::NativeBuffer>,
        _y_partial: impl BufferArg<'y_partial, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _t_count: u32,
        _d_model: u32,
        _k_input: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
