use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeCountsOffsetsFusedKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeCountsOffsetsFusedCpuKernel;

impl MoeCountsOffsetsFusedKernel for MoeCountsOffsetsFusedCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'topk_ids, 'offsets, 'sum_k_out, 'partials, 'encoder>(
        &self,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _sum_k_out: impl BufferArg<'sum_k_out, <Self::Backend as Backend>::NativeBuffer>,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _t_input: u32,
        _e_input: u32,
        _k_input: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'topk_ids, 'offsets, 'sum_k_out, 'partials, 'encoder, 'predicate>(
        &self,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _sum_k_out: impl BufferArg<'sum_k_out, <Self::Backend as Backend>::NativeBuffer>,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _t_input: u32,
        _e_input: u32,
        _k_input: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
