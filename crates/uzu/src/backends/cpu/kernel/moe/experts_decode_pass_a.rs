use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeExpertsDecodePassAKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeExpertsDecodePassACpuKernel;

impl MoeExpertsDecodePassAKernel for MoeExpertsDecodePassACpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _gating_sel: u32,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<
        'x_perm,
        'expert_offsets,
        'w13_all,
        'hidden_out,
        'up_biases,
        'tile_map,
        '__dsl_indirect_dispatch_buffer,
        'encoder,
    >(
        &self,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _w13_all: impl BufferArg<'w13_all, <Self::Backend as Backend>::NativeBuffer>,
        _hidden_out: impl BufferArg<'hidden_out, <Self::Backend as Backend>::NativeBuffer>,
        _up_biases: impl BufferArg<'up_biases, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
        _gate_clip_min: f32,
        _gate_clip_max: f32,
        _up_clip_min: f32,
        _up_clip_max: f32,
        _silu_alpha: f32,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        __dsl_indirect_dispatch_buffer: impl BufferArg<
            '__dsl_indirect_dispatch_buffer,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'x_perm,
        'expert_offsets,
        'w13_all,
        'hidden_out,
        'up_biases,
        'tile_map,
        '__dsl_indirect_dispatch_buffer,
        'encoder,
        'predicate,
    >(
        &self,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _w13_all: impl BufferArg<'w13_all, <Self::Backend as Backend>::NativeBuffer>,
        _hidden_out: impl BufferArg<'hidden_out, <Self::Backend as Backend>::NativeBuffer>,
        _up_biases: impl BufferArg<'up_biases, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
        _gate_clip_min: f32,
        _gate_clip_max: f32,
        _up_clip_min: f32,
        _up_clip_max: f32,
        _silu_alpha: f32,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        __dsl_indirect_dispatch_buffer: impl BufferArg<
            '__dsl_indirect_dispatch_buffer,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
