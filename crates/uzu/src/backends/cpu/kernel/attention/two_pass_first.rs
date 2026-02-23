use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{AttentionTwoPass1Kernel, BufferArg},
        },
        cpu::Cpu,
    },
};

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

    fn encode_if<'queries, 'keys, 'values, 'out, 'sums, 'maxs, 'fmask, 'sinks, 'encoder, 'predicate>(
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
