use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, ShortConvPackKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct ShortConvPackCpuKernel;

impl ShortConvPackKernel for ShortConvPackCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'state_in, 'in_proj, 'padded, 'encoder>(
        &self,
        _state_in: impl BufferArg<'state_in, <Self::Backend as Backend>::NativeBuffer>,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _state_stride: u32,
        _suffix_len: u32,
        _in_proj_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'state_in, 'in_proj, 'padded, 'encoder, 'predicate>(
        &self,
        _state_in: impl BufferArg<'state_in, <Self::Backend as Backend>::NativeBuffer>,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _state_stride: u32,
        _suffix_len: u32,
        _in_proj_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
