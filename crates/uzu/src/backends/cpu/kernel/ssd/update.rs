use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, SSDUpdateKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct SSDUpdateCpuKernel;

impl SSDUpdateKernel for SSDUpdateCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'x, 'dt_raw, 'b, 'c, 'd, 'z, 'state, 'y, 'next_state, 'encoder>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _dt_raw: impl BufferArg<'dt_raw, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _c: impl BufferArg<'c, <Self::Backend as Backend>::NativeBuffer>,
        _d: impl BufferArg<'d, <Self::Backend as Backend>::NativeBuffer>,
        _z: impl BufferArg<'z, <Self::Backend as Backend>::NativeBuffer>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _group_size: u32,
        _state_size: u32,
        _x_strides: &[u32],
        _dt_strides: &[u32],
        _cb_strides: &[u32],
        _state_strides: &[u32],
        _b_size: u32,
        _h_size: u32,
        _dh_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'x, 'dt_raw, 'b, 'c, 'd, 'z, 'state, 'y, 'next_state, 'encoder, 'predicate>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _dt_raw: impl BufferArg<'dt_raw, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _c: impl BufferArg<'c, <Self::Backend as Backend>::NativeBuffer>,
        _d: impl BufferArg<'d, <Self::Backend as Backend>::NativeBuffer>,
        _z: impl BufferArg<'z, <Self::Backend as Backend>::NativeBuffer>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _group_size: u32,
        _state_size: u32,
        _x_strides: &[u32],
        _dt_strides: &[u32],
        _cb_strides: &[u32],
        _state_strides: &[u32],
        _b_size: u32,
        _h_size: u32,
        _dh_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
