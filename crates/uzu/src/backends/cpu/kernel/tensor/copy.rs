use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, TensorCopyKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct TensorCopyCpuKernel;

impl TensorCopyKernel for TensorCopyCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'src_buffer, 'dst_buffer, 'encoder>(
        &self,
        _src_buffer: impl BufferArg<'src_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _dst_buffer: impl BufferArg<'dst_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'src_buffer, 'dst_buffer, 'encoder, 'predicate>(
        &self,
        _src_buffer: impl BufferArg<'src_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _dst_buffer: impl BufferArg<'dst_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
