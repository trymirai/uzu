use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeExpertsDecodeSinglePassAKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeExpertsDecodeSinglePassACpuKernel;

impl MoeExpertsDecodeSinglePassAKernel for MoeExpertsDecodeSinglePassACpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _gating_sel: u32,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'x, 'topk_ids, 'w13_all, 'biases, 'hidden_out, 'encoder>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _w13_all: impl BufferArg<'w13_all, <Self::Backend as Backend>::NativeBuffer>,
        _biases: impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>,
        _hidden_out: impl BufferArg<'hidden_out, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _k: u32,
        _silu_alpha: f32,
        _gate_clip_min: f32,
        _gate_clip_max: f32,
        _up_clip_min: f32,
        _up_clip_max: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'x, 'topk_ids, 'w13_all, 'biases, 'hidden_out, 'encoder, 'predicate>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _topk_ids: impl BufferArg<'topk_ids, <Self::Backend as Backend>::NativeBuffer>,
        _w13_all: impl BufferArg<'w13_all, <Self::Backend as Backend>::NativeBuffer>,
        _biases: impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>,
        _hidden_out: impl BufferArg<'hidden_out, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _k: u32,
        _silu_alpha: f32,
        _gate_clip_min: f32,
        _gate_clip_max: f32,
        _up_clip_min: f32,
        _up_clip_max: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
