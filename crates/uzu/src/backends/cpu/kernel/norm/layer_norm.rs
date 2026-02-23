use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, LayerNormKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct LayerNormCpuKernel;

impl LayerNormKernel for LayerNormCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _IN: DataType,
        _SC: DataType,
        _OUT: DataType,
        _ACC: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'scales, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _model_dim: u32,
        _epsilon: f32,
        _scale_offset: f32,
        _full_layer: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'scales, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _model_dim: u32,
        _epsilon: f32,
        _scale_offset: f32,
        _full_layer: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
