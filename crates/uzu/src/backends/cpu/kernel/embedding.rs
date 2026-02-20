use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, FullPrecisionEmbeddingLookupKernel, QuantizedEmbeddingLookupKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct FullPrecisionEmbeddingLookupCpuKernel;

impl FullPrecisionEmbeddingLookupKernel for FullPrecisionEmbeddingLookupCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'token_ids, 'weights, 'output, 'encoder>(
        &self,
        _token_ids: impl BufferArg<'token_ids, <Self::Backend as Backend>::NativeBuffer>,
        _weights: impl BufferArg<'weights, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _model_dim: u32,
        _input_scale: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'token_ids, 'weights, 'output, 'encoder, 'predicate>(
        &self,
        _token_ids: impl BufferArg<'token_ids, <Self::Backend as Backend>::NativeBuffer>,
        _weights: impl BufferArg<'weights, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _model_dim: u32,
        _input_scale: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct QuantizedEmbeddingLookupCpuKernel;

impl QuantizedEmbeddingLookupKernel for QuantizedEmbeddingLookupCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'token_ids, 'weights, 'scales, 'biases, 'output, 'encoder>(
        &self,
        _token_ids: impl BufferArg<'token_ids, <Self::Backend as Backend>::NativeBuffer>,
        _weights: impl BufferArg<'weights, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _biases: impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _model_dim: u32,
        _group_size: u32,
        _input_scale: f32,
        _quant_mode: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'token_ids, 'weights, 'scales, 'biases, 'output, 'encoder, 'predicate>(
        &self,
        _token_ids: impl BufferArg<'token_ids, <Self::Backend as Backend>::NativeBuffer>,
        _weights: impl BufferArg<'weights, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _biases: impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _model_dim: u32,
        _group_size: u32,
        _input_scale: f32,
        _quant_mode: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
