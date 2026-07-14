use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            Kernels, MoeExpertsDecodeDownFused2DKernel, MoeExpertsDecodePassAKernel, MoePassABuildRowMapKernel,
            MoePassABuildTileMapKernel, MoePassATileCountsKernel, MoePassATileScanKernel,
            MoePassAWriteDispatchArgsKernel,
        },
    },
    data_type::DataType,
    encodable_block::mlp::moe::experts_two_pass_prefill::MoeExpertsTwoPassArguments,
};

pub struct MoeExpertsTwoPassDecodeBlock<B: Backend> {
    counts: <B::Kernels as Kernels>::MoePassATileCountsKernel,
    scan: <B::Kernels as Kernels>::MoePassATileScanKernel,
    row_map: <B::Kernels as Kernels>::MoePassABuildRowMapKernel,
    build_map: <B::Kernels as Kernels>::MoePassABuildTileMapKernel,
    dispatch: <B::Kernels as Kernels>::MoePassAWriteDispatchArgsKernel,
    pass_a_indirect: <B::Kernels as Kernels>::MoeExpertsDecodePassAKernel,
    fused_down: <B::Kernels as Kernels>::MoeExpertsDecodeDownFused2DKernel,
    data_type: DataType,
}

impl<B: Backend> MoeExpertsTwoPassDecodeBlock<B> {
    pub fn new(
        ctx: &B::Context,
        data_type: DataType,
        gating_code: u32,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            counts: <B::Kernels as Kernels>::MoePassATileCountsKernel::new(ctx)?,
            scan: <B::Kernels as Kernels>::MoePassATileScanKernel::new(ctx)?,
            row_map: <B::Kernels as Kernels>::MoePassABuildRowMapKernel::new(ctx)?,
            build_map: <B::Kernels as Kernels>::MoePassABuildTileMapKernel::new(ctx)?,
            dispatch: <B::Kernels as Kernels>::MoePassAWriteDispatchArgsKernel::new(ctx)?,
            pass_a_indirect: <B::Kernels as Kernels>::MoeExpertsDecodePassAKernel::new(ctx, data_type, gating_code)?,
            fused_down: <B::Kernels as Kernels>::MoeExpertsDecodeDownFused2DKernel::new(ctx, data_type, DataType::F32)?,
            data_type,
        })
    }

    pub fn encode(
        &self,
        args: MoeExpertsTwoPassArguments<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        const BLOCK_M: usize = 4;
        let h_blocks = args.d_ff.div_ceil(BLOCK_M) as u32;

        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[args.num_routed_experts], DataType::U32))?;
        self.counts.encode(args.expert_offsets, &mut tile_counts, args.num_routed_experts as u32, h_blocks, encoder);

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
            encoder.allocate_scratch(size_for_shape(&[args.total_rows * h_blocks as usize * 3], DataType::U32))?;
        self.build_map.encode(
            args.expert_offsets,
            &tile_offsets,
            &row_expert_map,
            &mut tile_map,
            args.total_rows as u32,
            h_blocks,
            encoder,
        );

        let mut dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(&total_tiles, &mut dispatch_args, 1, encoder);

        let mut hidden = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_ff], DataType::F32))?;

        self.pass_a_indirect.encode(
            args.x_perm,
            args.expert_offsets,
            args.w13_all,
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
        self.fused_down.encode(
            &hidden,
            &row_expert_map,
            args.w2_all,
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
