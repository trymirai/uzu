use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, ShortConvPrefillKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct ShortConvPrefillCpuKernel;

impl ShortConvPrefillKernel for ShortConvPrefillCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'padded, 'in_proj, 'w, 'b, 'out, 'state_out, 'encoder>(
        &self,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _state_out: impl BufferArg<'state_out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'padded, 'in_proj, 'w, 'b, 'out, 'state_out, 'encoder, 'predicate>(
        &self,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _state_out: impl BufferArg<'state_out, <Self::Backend as Backend>::NativeBuffer>,
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
