use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, QKNormKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct QKNormCpuKernel;

impl QKNormKernel for QKNormCpuKernel {
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

    fn encode<'qkv_input, 'scales, 'qkv_output, 'encoder>(
        &self,
        _qkv_input: impl BufferArg<'qkv_input, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _qkv_output: impl BufferArg<'qkv_output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _num_q_heads: u32,
        _num_kv_heads: u32,
        _head_dim: u32,
        _epsilon: f32,
        _scale_offset: f32,
        _head_offset: u32,
        _head_count: u32,
        _full_layer: bool,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'qkv_input, 'scales, 'qkv_output, 'encoder, 'predicate>(
        &self,
        _qkv_input: impl BufferArg<'qkv_input, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _qkv_output: impl BufferArg<'qkv_output, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _num_q_heads: u32,
        _num_kv_heads: u32,
        _head_dim: u32,
        _epsilon: f32,
        _scale_offset: f32,
        _head_offset: u32,
        _head_count: u32,
        _full_layer: bool,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
