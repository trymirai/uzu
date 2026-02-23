use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{AttentionUpdateKVCacheKernel, BufferArg},
        },
        cpu::Cpu,
    },
};

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
