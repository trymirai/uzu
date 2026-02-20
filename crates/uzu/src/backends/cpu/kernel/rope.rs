use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, RopeKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct RopeCpuKernel;

impl RopeKernel for RopeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'qkv, 'cosines, 'sines, 'token_positions, 'rotated_queries, 'rotated_keys, 'encoder>(
        &self,
        _qkv: impl BufferArg<'qkv, <Self::Backend as Backend>::NativeBuffer>,
        _cosines: impl BufferArg<'cosines, <Self::Backend as Backend>::NativeBuffer>,
        _sines: impl BufferArg<'sines, <Self::Backend as Backend>::NativeBuffer>,
        _token_positions: impl BufferArg<'token_positions, <Self::Backend as Backend>::NativeBuffer>,
        _rotated_queries: impl BufferArg<'rotated_queries, <Self::Backend as Backend>::NativeBuffer>,
        _rotated_keys: impl BufferArg<'rotated_keys, <Self::Backend as Backend>::NativeBuffer>,
        _head_dim: u32,
        _num_heads: u32,
        _num_groups: u32,
        _suffix_length: u32,
        _max_sequence_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'qkv,
        'cosines,
        'sines,
        'token_positions,
        'rotated_queries,
        'rotated_keys,
        'encoder,
        'predicate,
    >(
        &self,
        _qkv: impl BufferArg<'qkv, <Self::Backend as Backend>::NativeBuffer>,
        _cosines: impl BufferArg<'cosines, <Self::Backend as Backend>::NativeBuffer>,
        _sines: impl BufferArg<'sines, <Self::Backend as Backend>::NativeBuffer>,
        _token_positions: impl BufferArg<'token_positions, <Self::Backend as Backend>::NativeBuffer>,
        _rotated_queries: impl BufferArg<'rotated_queries, <Self::Backend as Backend>::NativeBuffer>,
        _rotated_keys: impl BufferArg<'rotated_keys, <Self::Backend as Backend>::NativeBuffer>,
        _head_dim: u32,
        _num_heads: u32,
        _num_groups: u32,
        _suffix_length: u32,
        _max_sequence_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
