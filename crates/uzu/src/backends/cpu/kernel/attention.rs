use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                AttentionGemmKernel, AttentionSinglePassKernel, AttentionTwoPass1Kernel,
                AttentionTwoPass2Kernel, AttentionUpdateKVCacheKernel, BufferArg,
            },
        },
        cpu::backend::Cpu,
    },
};

pub struct AttentionGemmCpuKernel;

impl AttentionGemmKernel for AttentionGemmCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _BK: u32,
        _BD: u32,
        _align_q: bool,
        _align_k: bool,
        _do_causal: bool,
        _has_mask: bool,
        _has_sinks: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'q, 'k, 'v, 'o, 'mask, 'sinks, 'encoder>(
        &self,
        _q: impl BufferArg<'q, <Self::Backend as Backend>::NativeBuffer>,
        _k: impl BufferArg<'k, <Self::Backend as Backend>::NativeBuffer>,
        _v: impl BufferArg<'v, <Self::Backend as Backend>::NativeBuffer>,
        _o: impl BufferArg<'o, <Self::Backend as Backend>::NativeBuffer>,
        _params: crate::backends::common::gpu_types::attention::AttnParams,
        _mask_params: Option<crate::backends::common::gpu_types::attention::AttnMaskParams>,
        _mask: Option<impl BufferArg<'mask, <Self::Backend as Backend>::NativeBuffer>>,
        _sinks: Option<impl BufferArg<'sinks, <Self::Backend as Backend>::NativeBuffer>>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'q, 'k, 'v, 'o, 'mask, 'sinks, 'encoder, 'predicate>(
        &self,
        _q: impl BufferArg<'q, <Self::Backend as Backend>::NativeBuffer>,
        _k: impl BufferArg<'k, <Self::Backend as Backend>::NativeBuffer>,
        _v: impl BufferArg<'v, <Self::Backend as Backend>::NativeBuffer>,
        _o: impl BufferArg<'o, <Self::Backend as Backend>::NativeBuffer>,
        _params: crate::backends::common::gpu_types::attention::AttnParams,
        _mask_params: Option<crate::backends::common::gpu_types::attention::AttnMaskParams>,
        _mask: Option<impl BufferArg<'mask, <Self::Backend as Backend>::NativeBuffer>>,
        _sinks: Option<impl BufferArg<'sinks, <Self::Backend as Backend>::NativeBuffer>>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AttentionSinglePassCpuKernel;

impl AttentionSinglePassKernel for AttentionSinglePassCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _head_dim: u32,
        _float_mask: bool,
        _has_mask: bool,
        _has_sinks: bool,
        _do_causal: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'queries, 'keys, 'values, 'out, 'fmask, 'sinks, 'encoder>(
        &self,
        _queries: impl BufferArg<'queries, <Self::Backend as Backend>::NativeBuffer>,
        _keys: impl BufferArg<'keys, <Self::Backend as Backend>::NativeBuffer>,
        _values: impl BufferArg<'values, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _gqa_factor: u32,
        _sequence_length: u32,
        _k_head_stride: u32,
        _k_seq_stride: u32,
        _v_head_stride: u32,
        _v_seq_stride: u32,
        _scale: f32,
        _fmask: Option<impl BufferArg<'fmask, <Self::Backend as Backend>::NativeBuffer>>,
        _mask_kv_seq_stride: Option<u32>,
        _mask_q_seq_stride: Option<u32>,
        _mask_head_stride: Option<u32>,
        _sinks: Option<impl BufferArg<'sinks, <Self::Backend as Backend>::NativeBuffer>>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'queries, 'keys, 'values, 'out, 'fmask, 'sinks, 'encoder, 'predicate>(
        &self,
        _queries: impl BufferArg<'queries, <Self::Backend as Backend>::NativeBuffer>,
        _keys: impl BufferArg<'keys, <Self::Backend as Backend>::NativeBuffer>,
        _values: impl BufferArg<'values, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _gqa_factor: u32,
        _sequence_length: u32,
        _k_head_stride: u32,
        _k_seq_stride: u32,
        _v_head_stride: u32,
        _v_seq_stride: u32,
        _scale: f32,
        _fmask: Option<impl BufferArg<'fmask, <Self::Backend as Backend>::NativeBuffer>>,
        _mask_kv_seq_stride: Option<u32>,
        _mask_q_seq_stride: Option<u32>,
        _mask_head_stride: Option<u32>,
        _sinks: Option<impl BufferArg<'sinks, <Self::Backend as Backend>::NativeBuffer>>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AttentionTwoPass1CpuKernel;

impl AttentionTwoPass1Kernel for AttentionTwoPass1CpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _head_dim: u32,
        _float_mask: bool,
        _has_mask: bool,
        _has_sinks: bool,
        _do_causal: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'queries, 'keys, 'values, 'out, 'sums, 'maxs, 'fmask, 'sinks, 'encoder>(
        &self,
        _queries: impl BufferArg<'queries, <Self::Backend as Backend>::NativeBuffer>,
        _keys: impl BufferArg<'keys, <Self::Backend as Backend>::NativeBuffer>,
        _values: impl BufferArg<'values, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _sums: impl BufferArg<'sums, <Self::Backend as Backend>::NativeBuffer>,
        _maxs: impl BufferArg<'maxs, <Self::Backend as Backend>::NativeBuffer>,
        _gqa_factor: u32,
        _sequence_length: u32,
        _k_head_stride: u32,
        _k_seq_stride: u32,
        _v_head_stride: u32,
        _v_seq_stride: u32,
        _scale: f32,
        _num_heads: u32,
        _suffix_length: u32,
        _fmask: Option<impl BufferArg<'fmask, <Self::Backend as Backend>::NativeBuffer>>,
        _mask_kv_seq_stride: Option<u32>,
        _mask_q_seq_stride: Option<u32>,
        _mask_head_stride: Option<u32>,
        _sinks: Option<impl BufferArg<'sinks, <Self::Backend as Backend>::NativeBuffer>>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'queries,
        'keys,
        'values,
        'out,
        'sums,
        'maxs,
        'fmask,
        'sinks,
        'encoder,
        'predicate,
    >(
        &self,
        _queries: impl BufferArg<'queries, <Self::Backend as Backend>::NativeBuffer>,
        _keys: impl BufferArg<'keys, <Self::Backend as Backend>::NativeBuffer>,
        _values: impl BufferArg<'values, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _sums: impl BufferArg<'sums, <Self::Backend as Backend>::NativeBuffer>,
        _maxs: impl BufferArg<'maxs, <Self::Backend as Backend>::NativeBuffer>,
        _gqa_factor: u32,
        _sequence_length: u32,
        _k_head_stride: u32,
        _k_seq_stride: u32,
        _v_head_stride: u32,
        _v_seq_stride: u32,
        _scale: f32,
        _num_heads: u32,
        _suffix_length: u32,
        _fmask: Option<impl BufferArg<'fmask, <Self::Backend as Backend>::NativeBuffer>>,
        _mask_kv_seq_stride: Option<u32>,
        _mask_q_seq_stride: Option<u32>,
        _mask_head_stride: Option<u32>,
        _sinks: Option<impl BufferArg<'sinks, <Self::Backend as Backend>::NativeBuffer>>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AttentionTwoPass2CpuKernel;

impl AttentionTwoPass2Kernel for AttentionTwoPass2CpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _head_dim: u32,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'partials, 'sums, 'maxs, 'out, 'encoder>(
        &self,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _sums: impl BufferArg<'sums, <Self::Backend as Backend>::NativeBuffer>,
        _maxs: impl BufferArg<'maxs, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'partials, 'sums, 'maxs, 'out, 'encoder, 'predicate>(
        &self,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _sums: impl BufferArg<'sums, <Self::Backend as Backend>::NativeBuffer>,
        _maxs: impl BufferArg<'maxs, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AttentionUpdateKVCacheCpuKernel;

impl AttentionUpdateKVCacheKernel for AttentionUpdateKVCacheCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'rotated_keys, 'qkv, 'key_cache, 'value_cache, 'encoder>(
        &self,
        _rotated_keys: impl BufferArg<'rotated_keys, <Self::Backend as Backend>::NativeBuffer>,
        _qkv: impl BufferArg<'qkv, <Self::Backend as Backend>::NativeBuffer>,
        _key_cache: impl BufferArg<'key_cache, <Self::Backend as Backend>::NativeBuffer>,
        _value_cache: impl BufferArg<'value_cache, <Self::Backend as Backend>::NativeBuffer>,
        _num_groups: u32,
        _num_heads: u32,
        _head_dim: u32,
        _suffix_length: u32,
        _prefix_segment_length: u32,
        _max_sequence_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'rotated_keys, 'qkv, 'key_cache, 'value_cache, 'encoder, 'predicate>(
        &self,
        _rotated_keys: impl BufferArg<'rotated_keys, <Self::Backend as Backend>::NativeBuffer>,
        _qkv: impl BufferArg<'qkv, <Self::Backend as Backend>::NativeBuffer>,
        _key_cache: impl BufferArg<'key_cache, <Self::Backend as Backend>::NativeBuffer>,
        _value_cache: impl BufferArg<'value_cache, <Self::Backend as Backend>::NativeBuffer>,
        _num_groups: u32,
        _num_heads: u32,
        _head_dim: u32,
        _suffix_length: u32,
        _prefix_segment_length: u32,
        _max_sequence_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
