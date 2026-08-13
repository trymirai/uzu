use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{
            MoeBuildTileMapKernel, MoeExpertsMxfp4PrefillPassAKernel, MoeExpertsMxfp4PrefillPassBKernel,
            MoeTileCountsKernel, MoeTileScanKernel, MoeWriteDispatchArgsKernel,
        },
    },
    data_type::DataType,
    encodable_block::mlp::moe::experts_mxfp4_decode::MoeExpertsMxfp4Arguments,
};

/// Packed prefill schedule sized for GPT-OSS's sparse per-expert row occupancy.
pub struct MoeExpertsMxfp4PrefillBlock<B: Backend> {
    counts: <B::Kernels as Kernels>::MoeTileCountsKernel,
    scan: <B::Kernels as Kernels>::MoeTileScanKernel,
    build: <B::Kernels as Kernels>::MoeBuildTileMapKernel,
    dispatch: <B::Kernels as Kernels>::MoeWriteDispatchArgsKernel,
    pass_a: <B::Kernels as Kernels>::MoeExpertsMxfp4PrefillPassAKernel,
    pass_b: <B::Kernels as Kernels>::MoeExpertsMxfp4PrefillPassBKernel,
    data_type: DataType,
}

impl<B: Backend> MoeExpertsMxfp4PrefillBlock<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        gating_code: u32,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            counts: <B::Kernels as Kernels>::MoeTileCountsKernel::new(context)?,
            scan: <B::Kernels as Kernels>::MoeTileScanKernel::new(context)?,
            build: <B::Kernels as Kernels>::MoeBuildTileMapKernel::new(context)?,
            dispatch: <B::Kernels as Kernels>::MoeWriteDispatchArgsKernel::new(context)?,
            pass_a: <B::Kernels as Kernels>::MoeExpertsMxfp4PrefillPassAKernel::new(context, data_type, gating_code)?,
            pass_b: <B::Kernels as Kernels>::MoeExpertsMxfp4PrefillPassBKernel::new(context, data_type)?,
            data_type,
        })
    }

    pub fn encode(
        &self,
        args: MoeExpertsMxfp4Arguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[args.num_routed_experts], DataType::U32))?;
        const ROW_TILE: u32 = 16;
        self.counts.encode(args.expert_offsets, &mut tile_counts, args.num_routed_experts as u32, ROW_TILE, encoder);

        let mut tile_offsets =
            encoder.allocate_scratch(size_for_shape(&[args.num_routed_experts + 1], DataType::U32))?;
        let mut total_tiles = encoder.allocate_scratch(size_for_shape(&[8], DataType::U32))?;
        self.scan.encode(&tile_counts, &mut tile_offsets, &mut total_tiles, args.num_routed_experts as u32, encoder);

        let mut tile_map = encoder.allocate_scratch(size_for_shape(&[args.total_rows * 3], DataType::U32))?;
        self.build.encode(
            args.expert_offsets,
            &tile_offsets,
            &tile_counts,
            &mut tile_map,
            args.num_routed_experts as u32,
            ROW_TILE,
            encoder,
        );

        const PASS_A_COLUMNS: usize = 32;
        let pass_a_column_tiles = args.d_ff.div_ceil(PASS_A_COLUMNS) as u32;
        let mut pass_a_dispatch = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(&total_tiles, &mut pass_a_dispatch, pass_a_column_tiles, encoder);

        let mut hidden = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_ff], DataType::F32))?;
        self.pass_a.encode(
            args.x_perm,
            args.expert_offsets,
            args.w13_blocks,
            args.w13_scales,
            args.w13_global_scale,
            args.up_biases,
            &mut hidden,
            args.d_model as u32,
            args.d_ff as u32,
            args.num_routed_experts as u32,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            args.silu_alpha,
            &tile_map,
            &pass_a_dispatch,
            encoder,
        );

        const PASS_B_COLUMNS: usize = 32;
        let pass_b_column_tiles = args.d_model.div_ceil(PASS_B_COLUMNS) as u32;
        let mut pass_b_dispatch = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(&total_tiles, &mut pass_b_dispatch, pass_b_column_tiles, encoder);

        let mut output = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_model], self.data_type))?;
        self.pass_b.encode(
            &hidden,
            args.expert_offsets,
            args.w2_blocks,
            args.w2_scales,
            args.w2_global_scale,
            args.down_biases,
            &mut output,
            args.d_model as u32,
            args.d_ff as u32,
            args.num_routed_experts as u32,
            &tile_map,
            &pass_b_dispatch,
            encoder,
        );
        Ok(output)
    }
}
