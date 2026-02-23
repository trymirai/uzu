use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, ShortConvDecodeKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct ShortConvDecodeCpuKernel;

impl ShortConvDecodeKernel for ShortConvDecodeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'in_proj, 'w, 'b, 'state, 'out, 'next_state, 'encoder>(
        &self,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'in_proj, 'w, 'b, 'state, 'out, 'next_state, 'encoder, 'predicate>(
        &self,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
