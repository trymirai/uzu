use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, PoolingClsKernel, PoolingMeanKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct PoolingClsCpuKernel;

impl PoolingClsKernel for PoolingClsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _seq_len: u32,
        _hidden_dim: u32,
        _batch_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _seq_len: u32,
        _hidden_dim: u32,
        _batch_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct PoolingMeanCpuKernel;

impl PoolingMeanKernel for PoolingMeanCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _seq_len: u32,
        _hidden_dim: u32,
        _batch_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _seq_len: u32,
        _hidden_dim: u32,
        _batch_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
