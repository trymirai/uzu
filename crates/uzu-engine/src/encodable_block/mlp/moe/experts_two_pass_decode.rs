use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
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
        args: MoeExpertsTwoPassArguments<
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
        >,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        const BLOCK_M: u32 = 4;
        let h_blocks = args.d_ff.div_ceil(BLOCK_M);

        let mut tile_counts = command_buffer.allocate_scratch_for_shape(&[args.num_routed_experts], DataType::U32)?;
        self.counts.encode(args.expert_offsets, &mut tile_counts, args.num_routed_experts, h_blocks, command_buffer);

        let mut tile_offsets =
            command_buffer.allocate_scratch_for_shape(&[args.num_routed_experts + 1], DataType::U32)?;
        let mut total_tiles = command_buffer.allocate_scratch_for_shape(&[1], DataType::U32)?;
        self.scan.encode(&tile_counts, &mut tile_offsets, &mut total_tiles, args.num_routed_experts, command_buffer);

        let mut row_expert_map = command_buffer.allocate_scratch_for_shape(&[args.total_rows], DataType::U32)?;
        self.row_map.encode(
            args.expert_offsets,
            &mut row_expert_map,
            args.total_rows,
            args.num_routed_experts,
            command_buffer,
        );

        let mut tile_map = command_buffer.allocate_scratch_for_shape(&[args.total_rows, h_blocks, 3], DataType::U32)?;
        self.build_map.encode(
            args.expert_offsets,
            &tile_offsets,
            &row_expert_map,
            &mut tile_map,
            args.total_rows,
            h_blocks,
            command_buffer,
        );

        let mut dispatch_args = command_buffer.allocate_scratch_for_shape(&[3], DataType::U32)?;
        self.dispatch.encode(&total_tiles, &mut dispatch_args, 1, command_buffer);

        let mut hidden = command_buffer.allocate_scratch_for_shape(&[args.total_rows, args.d_ff], DataType::F32)?;

        self.pass_a_indirect.encode(
            args.x_perm,
            args.expert_offsets,
            args.w13_all,
            &mut hidden,
            args.up_biases,
            args.d_model,
            args.d_ff,
            args.num_routed_experts,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            args.silu_alpha,
            &tile_map,
            &dispatch_args,
            command_buffer,
        );

        let mut output = command_buffer.allocate_scratch_for_shape(&[args.total_rows, args.d_model], self.data_type)?;
        self.fused_down.encode(
            &hidden,
            &row_expert_map,
            args.w2_all,
            args.down_biases,
            &mut output,
            args.total_rows,
            args.d_model,
            args.d_ff,
            args.num_routed_experts,
            command_buffer,
        );

        Ok(output)
    }
}
