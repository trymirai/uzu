use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeRouterTopKKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeRouterTopKCpuKernel;

impl MoeRouterTopKKernel for MoeRouterTopKCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _ScalarT: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'weight, 'bias, 'topk_ids, 'topk_probs, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _weight: impl BufferArg<'weight, <Self::Backend as Backend>::NativeBuffer>,
        _bias: impl BufferArg<'bias, <Self::Backend as Backend>::NativeBuffer>,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _topk_probs: impl BufferArg<'topk_probs, <Self::Backend as Backend>::NativeBuffer>,
        _t: u32,
        _d_model: u32,
        _e: u32,
        _k: u32,
        _renorm: bool,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'weight, 'bias, 'topk_ids, 'topk_probs, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _weight: impl BufferArg<'weight, <Self::Backend as Backend>::NativeBuffer>,
        _bias: impl BufferArg<'bias, <Self::Backend as Backend>::NativeBuffer>,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _topk_probs: impl BufferArg<'topk_probs, <Self::Backend as Backend>::NativeBuffer>,
        _t: u32,
        _d_model: u32,
        _e: u32,
        _k: u32,
        _renorm: bool,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
