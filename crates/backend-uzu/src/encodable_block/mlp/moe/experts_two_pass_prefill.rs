use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{
            MoeBuildTileMapKernel, MoeExpertsPrefillPassAKernel, MoeExpertsPrefillPassBKernel, MoeTileCountsKernel,
            MoeTileScanKernel, MoeWriteDispatchArgsKernel,
        },
    },
    data_type::DataType,
};

pub struct MoeExpertsTwoPassPrefillBlock<B: Backend> {
    counts: <B::Kernels as Kernels>::MoeTileCountsKernel,
    scan: <B::Kernels as Kernels>::MoeTileScanKernel,
    build: <B::Kernels as Kernels>::MoeBuildTileMapKernel,
    dispatch: <B::Kernels as Kernels>::MoeWriteDispatchArgsKernel,
    pass_a_indirect: <B::Kernels as Kernels>::MoeExpertsPrefillPassAKernel,
    pass_b_indirect: <B::Kernels as Kernels>::MoeExpertsPrefillPassBKernel,
    data_type: DataType,
}

impl<B: Backend> MoeExpertsTwoPassPrefillBlock<B> {
    pub fn new(
        ctx: &B::Context,
        data_type: DataType,
        gating_code: u32,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            counts: <B::Kernels as Kernels>::MoeTileCountsKernel::new(ctx)?,
            scan: <B::Kernels as Kernels>::MoeTileScanKernel::new(ctx)?,
            build: <B::Kernels as Kernels>::MoeBuildTileMapKernel::new(ctx)?,
            dispatch: <B::Kernels as Kernels>::MoeWriteDispatchArgsKernel::new(ctx)?,
            pass_a_indirect: <B::Kernels as Kernels>::MoeExpertsPrefillPassAKernel::new(ctx, data_type, gating_code)?,
            pass_b_indirect: <B::Kernels as Kernels>::MoeExpertsPrefillPassBKernel::new(ctx, data_type)?,
            data_type,
        })
    }

    pub fn encode(
        &self,
        args: MoeExpertsTwoPassArguments<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[args.num_routed_experts], DataType::U32))?;
        self.counts.encode(args.expert_offsets, &mut tile_counts, args.num_routed_experts as u32, encoder);

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
            encoder,
        );

        const COL_TILE_FF: usize = 32; // Must match PASSA_BN in kernel
        let n_tiles_ff = args.d_ff.div_ceil(COL_TILE_FF) as u32;

        let mut pass_a_dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(&total_tiles, &mut pass_a_dispatch_args, n_tiles_ff, encoder);

        let mut hidden = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_ff], DataType::F32))?;
        encoder.encode_fill(&mut hidden, 0);

        self.pass_a_indirect.encode(
            args.x_perm,
            args.expert_offsets,
            args.w13_all,
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
            &pass_a_dispatch_args,
            encoder,
        );

        const COL_TILE_MODEL: usize = 64;
        let n_tiles_model = args.d_model.div_ceil(COL_TILE_MODEL) as u32;

        let mut pass_b_dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(&total_tiles, &mut pass_b_dispatch_args, n_tiles_model, encoder);

        let mut output = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_model], self.data_type))?;
        self.pass_b_indirect.encode(
            &hidden,
            args.expert_offsets,
            args.w2_all,
            args.down_biases,
            &mut output,
            args.d_model as u32,
            args.d_ff as u32,
            args.num_routed_experts as u32,
            &tile_map,
            &pass_b_dispatch_args,
            encoder,
        );
        Ok(output)
    }
}

pub struct MoeExpertsTwoPassArguments<'a, B: Backend> {
    pub x_perm: &'a Allocation<B>,
    pub expert_offsets: &'a Allocation<B>,
    pub w13_all: &'a Allocation<B>,
    pub w2_all: &'a Allocation<B>,
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
