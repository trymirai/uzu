use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                BufferArg, KVCacheUpdateKernel, LayerNormKernel, MaskUpdateKernel,
                MlpGateActMulKernel, QKNormKernel, RMSNormKernel,
            },
        },
        cpu::backend::Cpu,
    },
};

pub struct KVCacheUpdateCpuKernel;

impl KVCacheUpdateKernel for KVCacheUpdateCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'in_place_keys, 'in_place_values, 'encoder>(
        &self,
        _in_place_keys: impl BufferArg<'in_place_keys, <Self::Backend as Backend>::NativeBuffer>,
        _in_place_values: impl BufferArg<
            'in_place_values,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _swaps: &[crate::backends::common::gpu_types::kv_cache_update::Swap],
        _swap_count: u32,
        _num_heads: u32,
        _max_sequence_length: u32,
        _head_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'in_place_keys, 'in_place_values, 'encoder, 'predicate>(
        &self,
        _in_place_keys: impl BufferArg<'in_place_keys, <Self::Backend as Backend>::NativeBuffer>,
        _in_place_values: impl BufferArg<
            'in_place_values,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _swaps: &[crate::backends::common::gpu_types::kv_cache_update::Swap],
        _swap_count: u32,
        _num_heads: u32,
        _max_sequence_length: u32,
        _head_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

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

pub struct MaskUpdateCpuKernel;

impl MaskUpdateKernel for MaskUpdateCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'mask, 'encoder>(
        &self,
        _mask: impl BufferArg<'mask, <Self::Backend as Backend>::NativeBuffer>,
        _unmask_col: i32,
        _mask_col: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'mask, 'encoder, 'predicate>(
        &self,
        _mask: impl BufferArg<'mask, <Self::Backend as Backend>::NativeBuffer>,
        _unmask_col: i32,
        _mask_col: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MlpGateActMulCpuKernel;

impl MlpGateActMulKernel for MlpGateActMulCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'fused_up, 'hidden, 'encoder>(
        &self,
        _fused_up: impl BufferArg<'fused_up, <Self::Backend as Backend>::NativeBuffer>,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _h: i32,
        _m: i32,
        _act_type: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'fused_up, 'hidden, 'encoder, 'predicate>(
        &self,
        _fused_up: impl BufferArg<'fused_up, <Self::Backend as Backend>::NativeBuffer>,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _h: i32,
        _m: i32,
        _act_type: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

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
