use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, SplitInProjKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct SplitInProjCpuKernel;

impl SplitInProjKernel for SplitInProjCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'conv_out, 'z_out, 'dt_out, 'z_bias, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _conv_out: impl BufferArg<'conv_out, <Self::Backend as Backend>::NativeBuffer>,
        _z_out: impl BufferArg<'z_out, <Self::Backend as Backend>::NativeBuffer>,
        _dt_out: impl BufferArg<'dt_out, <Self::Backend as Backend>::NativeBuffer>,
        _z_bias: impl BufferArg<'z_bias, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_length: u32,
        _total_dim: u32,
        _conv_dim: u32,
        _inner_dim: u32,
        _num_heads: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'conv_out, 'z_out, 'dt_out, 'z_bias, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _conv_out: impl BufferArg<'conv_out, <Self::Backend as Backend>::NativeBuffer>,
        _z_out: impl BufferArg<'z_out, <Self::Backend as Backend>::NativeBuffer>,
        _dt_out: impl BufferArg<'dt_out, <Self::Backend as Backend>::NativeBuffer>,
        _z_bias: impl BufferArg<'z_bias, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_length: u32,
        _total_dim: u32,
        _conv_dim: u32,
        _inner_dim: u32,
        _num_heads: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
