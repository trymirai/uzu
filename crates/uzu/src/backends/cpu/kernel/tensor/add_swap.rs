use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, TensorAddSwapKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct TensorAddSwapCpuKernel;

impl TensorAddSwapKernel for TensorAddSwapCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'skip_buffer, 'main_buffer, 'encoder>(
        &self,
        _skip_buffer: impl BufferArg<'skip_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _main_buffer: impl BufferArg<'main_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'skip_buffer, 'main_buffer, 'encoder, 'predicate>(
        &self,
        _skip_buffer: impl BufferArg<'skip_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _main_buffer: impl BufferArg<'main_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
