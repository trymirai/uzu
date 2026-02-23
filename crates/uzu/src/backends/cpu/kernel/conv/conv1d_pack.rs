use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, Conv1dPackKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct Conv1dPackCpuKernel;

impl Conv1dPackKernel for Conv1dPackCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'state_in, 'x, 'padded, 'encoder>(
        &self,
        _state_in: impl BufferArg<'state_in, <Self::Backend as Backend>::NativeBuffer>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _state_stride: u32,
        _row_stride: u32,
        _suffix_len: u32,
        _num_channels: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'state_in, 'x, 'padded, 'encoder, 'predicate>(
        &self,
        _state_in: impl BufferArg<'state_in, <Self::Backend as Backend>::NativeBuffer>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _state_stride: u32,
        _row_stride: u32,
        _suffix_len: u32,
        _num_channels: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
