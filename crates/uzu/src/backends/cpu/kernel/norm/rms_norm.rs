use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, RMSNormKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct RMSNormCpuKernel;

impl RMSNormKernel for RMSNormCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _InputT: DataType,
        _ScaleT: DataType,
        _OutputT: DataType,
        _AccumT: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'scales, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _element_count: u32,
        _epsilon: f32,
        _scale_offset: f32,
        _full_layer: bool,
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
        _element_count: u32,
        _epsilon: f32,
        _scale_offset: f32,
        _full_layer: bool,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
