use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                BufferArg, MoeBuildTileMapKernel, MoeBlockBasesFromPartialsKernel,
                MoeCountsOffsetsFusedKernel, MoeExpertsDecodeDownFused2DKernel,
                MoeExpertsDecodePassAKernel, MoeExpertsDecodeSinglePassAKernel,
                MoeExpertsDecodeSinglePassBKernel, MoeExpertsPrefillPassAKernel,
                MoeExpertsPrefillPassBKernel, MoeFinalizeKernel, MoeGatherXPerm1DKernel,
                MoeGatherXPerm2DKernel, MoePassABuildRowMapKernel, MoePassABuildTileMapKernel,
                MoePassATileCountsKernel, MoePassATileScanKernel, MoePassAWriteDispatchArgsKernel,
                MoeRouterTopKKernel, MoeScatterBucketsKernel, MoeScatterBucketsMapKernel,
                MoeTileCountsKernel, MoeTileScanKernel, MoeWriteDispatchArgsKernel,
            },
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

pub struct MoeExpertsDecodeDownFused2DCpuKernel;

impl MoeExpertsDecodeDownFused2DKernel for MoeExpertsDecodeDownFused2DCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _AccumT: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'hidden, 'row_expert_map, 'w2_all, 'down_biases, 'y_out, 'encoder>(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _down_biases: impl BufferArg<'down_biases, <Self::Backend as Backend>::NativeBuffer>,
        _y_out: impl BufferArg<'y_out, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'hidden,
        'row_expert_map,
        'w2_all,
        'down_biases,
        'y_out,
        'encoder,
        'predicate,
    >(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _down_biases: impl BufferArg<'down_biases, <Self::Backend as Backend>::NativeBuffer>,
        _y_out: impl BufferArg<'y_out, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoeExpertsPrefillPassACpuKernel;

impl MoeExpertsPrefillPassAKernel for MoeExpertsPrefillPassACpuKernel {
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
        'up_biases,
        'hidden_out,
        'tile_map,
        '__dsl_indirect_dispatch_buffer,
        'encoder,
    >(
        &self,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _w13_all: impl BufferArg<'w13_all, <Self::Backend as Backend>::NativeBuffer>,
        _up_biases: impl BufferArg<'up_biases, <Self::Backend as Backend>::NativeBuffer>,
        _hidden_out: impl BufferArg<'hidden_out, <Self::Backend as Backend>::NativeBuffer>,
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
        'up_biases,
        'hidden_out,
        'tile_map,
        '__dsl_indirect_dispatch_buffer,
        'encoder,
        'predicate,
    >(
        &self,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _w13_all: impl BufferArg<'w13_all, <Self::Backend as Backend>::NativeBuffer>,
        _up_biases: impl BufferArg<'up_biases, <Self::Backend as Backend>::NativeBuffer>,
        _hidden_out: impl BufferArg<'hidden_out, <Self::Backend as Backend>::NativeBuffer>,
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

pub struct MoeExpertsPrefillPassBCpuKernel;

impl MoeExpertsPrefillPassBKernel for MoeExpertsPrefillPassBCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<
        'hidden,
        'expert_offsets,
        'w2_all,
        'down_biases,
        'output,
        'tile_map,
        '__dsl_indirect_dispatch_buffer,
        'encoder,
    >(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _down_biases: impl BufferArg<'down_biases, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
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
        'hidden,
        'expert_offsets,
        'w2_all,
        'down_biases,
        'output,
        'tile_map,
        '__dsl_indirect_dispatch_buffer,
        'encoder,
        'predicate,
    >(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _down_biases: impl BufferArg<'down_biases, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
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

pub struct MoeFinalizeCpuKernel;

impl MoeFinalizeKernel for MoeFinalizeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'tok2row, 'probs, 'y_partial, 'y, 'encoder>(
        &self,
        _tok2row: impl BufferArg<'tok2row, <Self::Backend as Backend>::NativeBuffer>,
        _probs: impl BufferArg<'probs, <Self::Backend as Backend>::NativeBuffer>,
        _y_partial: impl BufferArg<'y_partial, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _t_count: u32,
        _d_model: u32,
        _k_input: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'tok2row, 'probs, 'y_partial, 'y, 'encoder, 'predicate>(
        &self,
        _tok2row: impl BufferArg<'tok2row, <Self::Backend as Backend>::NativeBuffer>,
        _probs: impl BufferArg<'probs, <Self::Backend as Backend>::NativeBuffer>,
        _y_partial: impl BufferArg<'y_partial, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _t_count: u32,
        _d_model: u32,
        _k_input: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoeGatherXPerm2DCpuKernel;

impl MoeGatherXPerm2DKernel for MoeGatherXPerm2DCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'x, 'bucketed_ids, 'x_perm, 'sumk_buf, 'encoder>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _bucketed_ids: impl BufferArg<'bucketed_ids, <Self::Backend as Backend>::NativeBuffer>,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _sumk_buf: impl BufferArg<'sumk_buf, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _t: u32,
        _k: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'x, 'bucketed_ids, 'x_perm, 'sumk_buf, 'encoder, 'predicate>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _bucketed_ids: impl BufferArg<'bucketed_ids, <Self::Backend as Backend>::NativeBuffer>,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _sumk_buf: impl BufferArg<'sumk_buf, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _t: u32,
        _k: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoeGatherXPerm1DCpuKernel;

impl MoeGatherXPerm1DKernel for MoeGatherXPerm1DCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'x, 'bucketed_ids, 'x_perm, 'sumk_buf, 'encoder>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _bucketed_ids: impl BufferArg<'bucketed_ids, <Self::Backend as Backend>::NativeBuffer>,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _sumk_buf: impl BufferArg<'sumk_buf, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _t: u32,
        _k: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'x, 'bucketed_ids, 'x_perm, 'sumk_buf, 'encoder, 'predicate>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _bucketed_ids: impl BufferArg<'bucketed_ids, <Self::Backend as Backend>::NativeBuffer>,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _sumk_buf: impl BufferArg<'sumk_buf, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _t: u32,
        _k: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

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

pub struct MoeBlockBasesFromPartialsCpuKernel;

impl MoeBlockBasesFromPartialsKernel for MoeBlockBasesFromPartialsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'partials, 'block_bases, 'block_alloc, 'encoder>(
        &self,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _block_bases: impl BufferArg<'block_bases, <Self::Backend as Backend>::NativeBuffer>,
        _block_alloc: impl BufferArg<'block_alloc, <Self::Backend as Backend>::NativeBuffer>,
        _e_input: u32,
        _num_blocks: u32,
        _num_tiles: u32,
        _capacity_per_expert: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'partials, 'block_bases, 'block_alloc, 'encoder, 'predicate>(
        &self,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _block_bases: impl BufferArg<'block_bases, <Self::Backend as Backend>::NativeBuffer>,
        _block_alloc: impl BufferArg<'block_alloc, <Self::Backend as Backend>::NativeBuffer>,
        _e_input: u32,
        _num_blocks: u32,
        _num_tiles: u32,
        _capacity_per_expert: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoeScatterBucketsCpuKernel;

impl MoeScatterBucketsKernel for MoeScatterBucketsCpuKernel {
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
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

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

pub struct MoeTileCountsCpuKernel;

impl MoeTileCountsKernel for MoeTileCountsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'offsets, 'tile_counts, 'encoder>(
        &self,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'offsets, 'tile_counts, 'encoder, 'predicate>(
        &self,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoeTileScanCpuKernel;

impl MoeTileScanKernel for MoeTileScanCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'tile_counts, 'tile_row_offsets, 'total_tiles_buf, 'encoder>(
        &self,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_row_offsets: impl BufferArg<
            'tile_row_offsets,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _total_tiles_buf: impl BufferArg<'total_tiles_buf, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'tile_counts, 'tile_row_offsets, 'total_tiles_buf, 'encoder, 'predicate>(
        &self,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_row_offsets: impl BufferArg<
            'tile_row_offsets,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _total_tiles_buf: impl BufferArg<'total_tiles_buf, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoeBuildTileMapCpuKernel;

impl MoeBuildTileMapKernel for MoeBuildTileMapCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'offsets, 'tile_row_offsets, 'tile_counts, 'tile_map, 'encoder>(
        &self,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_row_offsets: impl BufferArg<
            'tile_row_offsets,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'offsets, 'tile_row_offsets, 'tile_counts, 'tile_map, 'encoder, 'predicate>(
        &self,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_row_offsets: impl BufferArg<
            'tile_row_offsets,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoeWriteDispatchArgsCpuKernel;

impl MoeWriteDispatchArgsKernel for MoeWriteDispatchArgsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'total_tiles_buf, 'dispatch_args, 'encoder>(
        &self,
        _total_tiles_buf: impl BufferArg<'total_tiles_buf, <Self::Backend as Backend>::NativeBuffer>,
        _dispatch_args: impl BufferArg<'dispatch_args, <Self::Backend as Backend>::NativeBuffer>,
        _num_tiles_n: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'total_tiles_buf, 'dispatch_args, 'encoder, 'predicate>(
        &self,
        _total_tiles_buf: impl BufferArg<'total_tiles_buf, <Self::Backend as Backend>::NativeBuffer>,
        _dispatch_args: impl BufferArg<'dispatch_args, <Self::Backend as Backend>::NativeBuffer>,
        _num_tiles_n: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoePassATileCountsCpuKernel;

impl MoePassATileCountsKernel for MoePassATileCountsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'expert_offsets, 'tile_counts, 'encoder>(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _h_blocks: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'expert_offsets, 'tile_counts, 'encoder, 'predicate>(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _h_blocks: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoePassATileScanCpuKernel;

impl MoePassATileScanKernel for MoePassATileScanCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'tile_counts, 'tile_offsets, 'total_tiles, 'encoder>(
        &self,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_offsets: impl BufferArg<'tile_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _total_tiles: impl BufferArg<'total_tiles, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'tile_counts, 'tile_offsets, 'total_tiles, 'encoder, 'predicate>(
        &self,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_offsets: impl BufferArg<'tile_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _total_tiles: impl BufferArg<'total_tiles, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoePassABuildRowMapCpuKernel;

impl MoePassABuildRowMapKernel for MoePassABuildRowMapCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'expert_offsets, 'row_expert_map, 'encoder>(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'expert_offsets, 'row_expert_map, 'encoder, 'predicate>(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoePassABuildTileMapCpuKernel;

impl MoePassABuildTileMapKernel for MoePassABuildTileMapCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'expert_offsets, 'tile_offsets, 'row_expert_map, 'tile_map, 'encoder>(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_offsets: impl BufferArg<'tile_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _h_blocks: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'expert_offsets,
        'tile_offsets,
        'row_expert_map,
        'tile_map,
        'encoder,
        'predicate,
    >(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_offsets: impl BufferArg<'tile_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _h_blocks: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MoePassAWriteDispatchArgsCpuKernel;

impl MoePassAWriteDispatchArgsKernel for MoePassAWriteDispatchArgsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'total_tiles, 'dispatch_args, 'encoder>(
        &self,
        _total_tiles: impl BufferArg<'total_tiles, <Self::Backend as Backend>::NativeBuffer>,
        _dispatch_args: impl BufferArg<'dispatch_args, <Self::Backend as Backend>::NativeBuffer>,
        _num_tiles_y: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'total_tiles, 'dispatch_args, 'encoder, 'predicate>(
        &self,
        _total_tiles: impl BufferArg<'total_tiles, <Self::Backend as Backend>::NativeBuffer>,
        _dispatch_args: impl BufferArg<'dispatch_args, <Self::Backend as Backend>::NativeBuffer>,
        _num_tiles_y: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
