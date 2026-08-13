use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            Kernels, MoeExpertsMxfp4DecodeDownFused2DKernel, MoeExpertsMxfp4DecodePassAKernel,
            MoePassABuildRowMapKernel, MoePassABuildTileMapKernel, MoePassATileCountsKernel, MoePassATileScanKernel,
            MoePassAWriteDispatchArgsKernel,
        },
    },
    data_type::DataType,
};

/// Correctness-first decode path for native MXFP4 expert projections.
pub struct MoeExpertsMxfp4DecodeBlock<B: Backend> {
    counts: <B::Kernels as Kernels>::MoePassATileCountsKernel,
    scan: <B::Kernels as Kernels>::MoePassATileScanKernel,
    row_map: <B::Kernels as Kernels>::MoePassABuildRowMapKernel,
    build_map: <B::Kernels as Kernels>::MoePassABuildTileMapKernel,
    dispatch: <B::Kernels as Kernels>::MoePassAWriteDispatchArgsKernel,
    pass_a: <B::Kernels as Kernels>::MoeExpertsMxfp4DecodePassAKernel,
    down: <B::Kernels as Kernels>::MoeExpertsMxfp4DecodeDownFused2DKernel,
    data_type: DataType,
}

impl<B: Backend> MoeExpertsMxfp4DecodeBlock<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        gating_code: u32,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            counts: <B::Kernels as Kernels>::MoePassATileCountsKernel::new(context)?,
            scan: <B::Kernels as Kernels>::MoePassATileScanKernel::new(context)?,
            row_map: <B::Kernels as Kernels>::MoePassABuildRowMapKernel::new(context)?,
            build_map: <B::Kernels as Kernels>::MoePassABuildTileMapKernel::new(context)?,
            dispatch: <B::Kernels as Kernels>::MoePassAWriteDispatchArgsKernel::new(context)?,
            pass_a: <B::Kernels as Kernels>::MoeExpertsMxfp4DecodePassAKernel::new(context, data_type, gating_code)?,
            down: <B::Kernels as Kernels>::MoeExpertsMxfp4DecodeDownFused2DKernel::new(
                context,
                data_type,
                DataType::F32,
            )?,
            data_type,
        })
    }

    pub fn encode(
        &self,
        args: MoeExpertsMxfp4Arguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        const HIDDEN_VALUES_PER_TILE: usize = 8;
        let hidden_blocks = args.d_ff.div_ceil(HIDDEN_VALUES_PER_TILE) as u32;

        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[args.num_routed_experts], DataType::U32))?;
        self.counts.encode(
            args.expert_offsets,
            &mut tile_counts,
            args.num_routed_experts as u32,
            hidden_blocks,
            encoder,
        );

        let mut tile_offsets =
            encoder.allocate_scratch(size_for_shape(&[args.num_routed_experts + 1], DataType::U32))?;
        let mut total_tiles = encoder.allocate_scratch(size_for_shape(&[1], DataType::U32))?;
        self.scan.encode(&tile_counts, &mut tile_offsets, &mut total_tiles, args.num_routed_experts as u32, encoder);

        let mut row_expert_map = encoder.allocate_scratch(size_for_shape(&[args.total_rows], DataType::U32))?;
        self.row_map.encode(
            args.expert_offsets,
            &mut row_expert_map,
            args.total_rows as u32,
            args.num_routed_experts as u32,
            encoder,
        );

        let mut tile_map =
            encoder.allocate_scratch(size_for_shape(&[args.total_rows * hidden_blocks as usize * 3], DataType::U32))?;
        self.build_map.encode(
            args.expert_offsets,
            &tile_offsets,
            &row_expert_map,
            &mut tile_map,
            args.total_rows as u32,
            hidden_blocks,
            encoder,
        );

        let mut dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(&total_tiles, &mut dispatch_args, 1, encoder);

        let mut hidden = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_ff], DataType::F32))?;
        self.pass_a.encode(
            args.x_perm,
            args.expert_offsets,
            args.w13_blocks,
            args.w13_scales,
            args.w13_global_scale,
            &mut hidden,
            args.up_biases,
            args.d_model as u32,
            args.d_ff as u32,
            args.num_routed_experts as u32,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            args.silu_alpha,
            &tile_map,
            &dispatch_args,
            encoder,
        );

        let mut output = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_model], self.data_type))?;
        self.down.encode(
            &hidden,
            &row_expert_map,
            args.w2_blocks,
            args.w2_scales,
            args.w2_global_scale,
            args.down_biases,
            &mut output,
            args.total_rows as u32,
            args.d_model as u32,
            args.d_ff as u32,
            args.num_routed_experts as u32,
            encoder,
        );

        Ok(output)
    }
}

/// Canonical Lalamo experts: W13 scales cover 16 values, while W2 scales cover 32.
pub struct MoeExpertsMxfp4Arguments<'a, B: Backend> {
    pub x_perm: &'a Allocation<B>,
    pub expert_offsets: &'a Allocation<B>,
    pub w13_blocks: &'a Allocation<B>,
    pub w13_scales: &'a Allocation<B>,
    pub w13_global_scale: &'a Allocation<B>,
    pub w2_blocks: &'a Allocation<B>,
    pub w2_scales: &'a Allocation<B>,
    pub w2_global_scale: &'a Allocation<B>,
    pub up_biases: &'a Allocation<B>,
    pub down_biases: &'a Allocation<B>,
    pub total_rows: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_routed_experts: usize,
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
}
