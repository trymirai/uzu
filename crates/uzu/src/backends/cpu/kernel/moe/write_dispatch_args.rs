use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeWriteDispatchArgsKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeWriteDispatchArgsCpuKernel;

impl MoeWriteDispatchArgsKernel for MoeWriteDispatchArgsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'total_tiles_buf, 'dispatch_args, 'encoder>(
        &self,
        _total_tiles_buf: impl BufferArg<'total_tiles_buf, <Self::Backend as Backend>::NativeBuffer>,
        _dispatch_args: impl BufferArg<'dispatch_args, <Self::Backend as Backend>::NativeBuffer>,
        _num_tiles_n: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'total_tiles_buf, 'dispatch_args, 'encoder, 'predicate>(
        &self,
        _total_tiles_buf: impl BufferArg<'total_tiles_buf, <Self::Backend as Backend>::NativeBuffer>,
        _dispatch_args: impl BufferArg<'dispatch_args, <Self::Backend as Backend>::NativeBuffer>,
        _num_tiles_n: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
