use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, Conv1dDecodeKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct Conv1dDecodeCpuKernel;

impl Conv1dDecodeKernel for Conv1dDecodeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _activation_type: u32,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'x, 'w, 'b, 'state, 'x_out, 'b_out, 'c_out, 'next_state, 'encoder>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _x_out: impl BufferArg<'x_out, <Self::Backend as Backend>::NativeBuffer>,
        _b_out: impl BufferArg<'b_out, <Self::Backend as Backend>::NativeBuffer>,
        _c_out: impl BufferArg<'c_out, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _kernel_size: u32,
        _row_stride: u32,
        _state_stride: u32,
        _num_channels: u32,
        _suffix_len: u32,
        _inner_dim: u32,
        _proj_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'x,
        'w,
        'b,
        'state,
        'x_out,
        'b_out,
        'c_out,
        'next_state,
        'encoder,
        'predicate,
    >(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _x_out: impl BufferArg<'x_out, <Self::Backend as Backend>::NativeBuffer>,
        _b_out: impl BufferArg<'b_out, <Self::Backend as Backend>::NativeBuffer>,
        _c_out: impl BufferArg<'c_out, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _kernel_size: u32,
        _row_stride: u32,
        _state_stride: u32,
        _num_channels: u32,
        _suffix_len: u32,
        _inner_dim: u32,
        _proj_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
