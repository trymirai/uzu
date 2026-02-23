use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeScatterBucketsMapKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeScatterBucketsMapCpuKernel;

impl MoeScatterBucketsMapKernel for MoeScatterBucketsMapCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<
        'topk_ids,
        'topk_probs,
        'offsets,
        'block_bases,
        'block_alloc,
        'out_ids,
        'out_probs,
        'tok2row,
        'encoder,
    >(
        &self,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _topk_probs: impl BufferArg<'topk_probs, <Self::Backend as Backend>::NativeBuffer>,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _block_bases: impl BufferArg<'block_bases, <Self::Backend as Backend>::NativeBuffer>,
        _block_alloc: impl BufferArg<'block_alloc, <Self::Backend as Backend>::NativeBuffer>,
        _out_ids: impl BufferArg<'out_ids, <Self::Backend as Backend>::NativeBuffer>,
        _out_probs: impl BufferArg<'out_probs, <Self::Backend as Backend>::NativeBuffer>,
        _t: u32,
        _e: u32,
        _k: u32,
        _num_blocks: u32,
        _num_tiles: u32,
        _tok2row: impl BufferArg<'tok2row, <Self::Backend as Backend>::NativeBuffer>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'topk_ids,
        'topk_probs,
        'offsets,
        'block_bases,
        'block_alloc,
        'out_ids,
        'out_probs,
        'tok2row,
        'encoder,
        'predicate,
    >(
        &self,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _topk_probs: impl BufferArg<'topk_probs, <Self::Backend as Backend>::NativeBuffer>,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _block_bases: impl BufferArg<'block_bases, <Self::Backend as Backend>::NativeBuffer>,
        _block_alloc: impl BufferArg<'block_alloc, <Self::Backend as Backend>::NativeBuffer>,
        _out_ids: impl BufferArg<'out_ids, <Self::Backend as Backend>::NativeBuffer>,
        _out_probs: impl BufferArg<'out_probs, <Self::Backend as Backend>::NativeBuffer>,
        _t: u32,
        _e: u32,
        _k: u32,
        _num_blocks: u32,
        _num_tiles: u32,
        _tok2row: impl BufferArg<'tok2row, <Self::Backend as Backend>::NativeBuffer>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
