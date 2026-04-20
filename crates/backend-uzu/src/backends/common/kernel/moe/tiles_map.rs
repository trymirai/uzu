use crate::backends::common::{
    Allocation, Backend, Encoder, Kernels,
    kernel::{MoeBuildTileMapKernel, MoeTileCountsKernel, MoeTileScanKernel, MoeWriteDispatchArgsKernel},
};

pub struct MoeTileMapKernels<B: Backend> {
    counts: <B::Kernels as Kernels>::MoeTileCountsKernel,
    scan: <B::Kernels as Kernels>::MoeTileScanKernel,
    build: <B::Kernels as Kernels>::MoeBuildTileMapKernel,
    dispatch: <B::Kernels as Kernels>::MoeWriteDispatchArgsKernel,
}

impl<B: Backend> MoeTileMapKernels<B> {
    pub fn new(ctx: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
            counts: <B::Kernels as Kernels>::MoeTileCountsKernel::new(ctx)?,
            scan: <B::Kernels as Kernels>::MoeTileScanKernel::new(ctx)?,
            build: <B::Kernels as Kernels>::MoeBuildTileMapKernel::new(ctx)?,
            dispatch: <B::Kernels as Kernels>::MoeWriteDispatchArgsKernel::new(ctx)?,
        })
    }

    pub fn encode_counts(
        &self,
        encoder: &mut Encoder<B>,
        args: MoeTileCountsArguments<B>,
    ) {
        self.counts.encode(args.offsets, args.tile_counts, args.e as u32, encoder);
    }

    pub fn encode_scan(
        &self,
        encoder: &mut Encoder<B>,
        args: MoeTileScanArguments<B>,
    ) {
        self.scan.encode(args.tile_counts, args.tile_offsets, args.total_tiles, args.e as u32, encoder);
    }

    pub fn encode_build_map(
        &self,
        encoder: &mut Encoder<B>,
        args: MoeTileMapBuildArguments<B>,
    ) {
        self.build.encode(
            args.expert_offsets,
            args.tile_offsets,
            args.tile_counts,
            args.tile_map,
            args.e as u32,
            encoder,
        );
    }

    pub fn encode_dispatch_args(
        &self,
        encoder: &mut Encoder<B>,
        args: MoeTileDispatchArguments<B>,
    ) {
        self.dispatch.encode(args.total_tiles, args.dispatch_args, args.num_tiles_x, encoder);
    }
}

pub struct MoeTileCountsArguments<'a, B: Backend> {
    pub offsets: &'a Allocation<B>,         // [E+1]
    pub tile_counts: &'a mut Allocation<B>, // [E]
    pub e: usize,
}

pub struct MoeTileScanArguments<'a, B: Backend> {
    pub tile_counts: &'a Allocation<B>,      // [E]
    pub tile_offsets: &'a mut Allocation<B>, // [E+1]
    pub total_tiles: &'a mut Allocation<B>,  // [>=2]
    pub e: usize,
}

pub struct MoeTileMapBuildArguments<'a, B: Backend> {
    pub expert_offsets: &'a Allocation<B>, // [E+1]
    pub tile_offsets: &'a Allocation<B>,   // [E+1]
    pub tile_counts: &'a Allocation<B>,    // [E]
    pub tile_map: &'a mut Allocation<B>,   // [total_tiles * 3]
    pub e: usize,
}

pub struct MoeTileDispatchArguments<'a, B: Backend> {
    pub total_tiles: &'a Allocation<B>,       // [>=1]
    pub dispatch_args: &'a mut Allocation<B>, // [3]
    pub num_tiles_x: u32,                     // x dimension for indirect dispatch
}
