use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, KVCacheUpdateKernel},
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
