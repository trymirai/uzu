use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeExpertsDecodeSinglePassBKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeExpertsDecodeSinglePassBCpuKernel;

impl MoeExpertsDecodeSinglePassBKernel for MoeExpertsDecodeSinglePassBCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'hidden, 'topk_ids, 'topk_probs, 'w2_all, 'biases, 'y, 'encoder>(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _topk_probs: impl BufferArg<'topk_probs, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _biases: impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _k_input: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'hidden, 'topk_ids, 'topk_probs, 'w2_all, 'biases, 'y, 'encoder, 'predicate>(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _topk_probs: impl BufferArg<'topk_probs, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _biases: impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _k_input: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
