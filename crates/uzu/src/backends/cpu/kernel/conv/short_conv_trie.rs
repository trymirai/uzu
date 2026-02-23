use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, ShortConvTrieKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct ShortConvTrieCpuKernel;

impl ShortConvTrieKernel for ShortConvTrieCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'in_proj, 'w, 'b, 'base_state, 'parents, 'out, 'suffix_state, 'encoder>(
        &self,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _base_state: impl BufferArg<'base_state, <Self::Backend as Backend>::NativeBuffer>,
        _parents: impl BufferArg<'parents, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_state: impl BufferArg<'suffix_state, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'in_proj,
        'w,
        'b,
        'base_state,
        'parents,
        'out,
        'suffix_state,
        'encoder,
        'predicate,
    >(
        &self,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _base_state: impl BufferArg<'base_state, <Self::Backend as Backend>::NativeBuffer>,
        _parents: impl BufferArg<'parents, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_state: impl BufferArg<'suffix_state, <Self::Backend as Backend>::NativeBuffer>,
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
