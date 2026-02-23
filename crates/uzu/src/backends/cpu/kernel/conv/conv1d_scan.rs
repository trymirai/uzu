use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, Conv1dScanKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct Conv1dScanCpuKernel;

impl Conv1dScanKernel for Conv1dScanCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _activation_type: u32,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'padded, 'w, 'b, 'x_out, 'b_out, 'c_out, 'state_out, 'encoder>(
        &self,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _x_out: impl BufferArg<'x_out, <Self::Backend as Backend>::NativeBuffer>,
        _b_out: impl BufferArg<'b_out, <Self::Backend as Backend>::NativeBuffer>,
        _c_out: impl BufferArg<'c_out, <Self::Backend as Backend>::NativeBuffer>,
        _state_out: impl BufferArg<'state_out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _row_stride: u32,
        _state_stride: u32,
        _num_channels: u32,
        _inner_dim: u32,
        _proj_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'padded, 'w, 'b, 'x_out, 'b_out, 'c_out, 'state_out, 'encoder, 'predicate>(
        &self,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _x_out: impl BufferArg<'x_out, <Self::Backend as Backend>::NativeBuffer>,
        _b_out: impl BufferArg<'b_out, <Self::Backend as Backend>::NativeBuffer>,
        _c_out: impl BufferArg<'c_out, <Self::Backend as Backend>::NativeBuffer>,
        _state_out: impl BufferArg<'state_out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _row_stride: u32,
        _state_stride: u32,
        _num_channels: u32,
        _inner_dim: u32,
        _proj_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
