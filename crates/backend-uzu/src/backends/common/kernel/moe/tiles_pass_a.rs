use crate::backends::common::{
    Allocation, Backend, Encoder,
    kernel::{
        Kernels, MoePassABuildRowMapKernel, MoePassABuildTileMapKernel, MoePassATileCountsKernel,
        MoePassATileScanKernel, MoePassAWriteDispatchArgsKernel,
    },
};

pub struct MoePassATileCountsArguments<'a, B: Backend> {
    pub expert_offsets: &'a Allocation<B>,  // [E+1]
    pub tile_counts: &'a mut Allocation<B>, // [E]
    pub e: usize,
    pub h_blocks: u32,
}

pub struct MoePassATileScanArguments<'a, B: Backend> {
    pub tile_counts: &'a Allocation<B>,      // [E]
    pub tile_offsets: &'a mut Allocation<B>, // [E+1]
    pub total_tiles: &'a mut Allocation<B>,  // [>=1]
    pub e: usize,
}

pub struct MoePassARowMapArguments<'a, B: Backend> {
    pub expert_offsets: &'a Allocation<B>,     // [E+1]
    pub row_expert_map: &'a mut Allocation<B>, // [total_rows]
    pub total_rows: usize,
    pub e: usize,
}

pub struct MoePassATileBuildArguments<'a, B: Backend> {
    pub expert_offsets: &'a Allocation<B>, // [E+1]
    pub tile_offsets: &'a Allocation<B>,   // [E+1]
    pub row_expert_map: &'a Allocation<B>, // [total_rows]
    pub tile_map: &'a mut Allocation<B>,   // [total_tiles * 3]
    pub total_rows: usize,
    pub h_blocks: u32,
}

pub struct MoePassATileDispatchArguments<'a, B: Backend> {
    pub total_tiles: &'a Allocation<B>,       // [>=1]
    pub dispatch_args: &'a mut Allocation<B>, // [3]
    pub num_tiles_y: u32,
}

pub struct MoePassATileKernels<B: Backend> {
    counts: <B::Kernels as Kernels>::MoePassATileCountsKernel,
    scan: <B::Kernels as Kernels>::MoePassATileScanKernel,
    row_map: <B::Kernels as Kernels>::MoePassABuildRowMapKernel,
    build_map: <B::Kernels as Kernels>::MoePassABuildTileMapKernel,
    dispatch: <B::Kernels as Kernels>::MoePassAWriteDispatchArgsKernel,
}

impl<B: Backend> MoePassATileKernels<B> {
    pub fn new(ctx: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
            counts: <B::Kernels as Kernels>::MoePassATileCountsKernel::new(ctx)?,
            scan: <B::Kernels as Kernels>::MoePassATileScanKernel::new(ctx)?,
            row_map: <B::Kernels as Kernels>::MoePassABuildRowMapKernel::new(ctx)?,
            build_map: <B::Kernels as Kernels>::MoePassABuildTileMapKernel::new(ctx)?,
            dispatch: <B::Kernels as Kernels>::MoePassAWriteDispatchArgsKernel::new(ctx)?,
        })
    }

    pub fn encode_counts(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassATileCountsArguments<B>,
    ) {
        self.counts.encode(args.expert_offsets, args.tile_counts, args.e as u32, args.h_blocks, encoder);
    }

    pub fn encode_scan(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassATileScanArguments<B>,
    ) {
        self.scan.encode(args.tile_counts, args.tile_offsets, args.total_tiles, args.e as u32, encoder);
    }

    pub fn encode_row_map(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassARowMapArguments<B>,
    ) {
        self.row_map.encode(args.expert_offsets, args.row_expert_map, args.total_rows as u32, args.e as u32, encoder);
    }

    pub fn encode_build_map(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassATileBuildArguments<B>,
    ) {
        self.build_map.encode(
            args.expert_offsets,
            args.tile_offsets,
            args.row_expert_map,
            args.tile_map,
            args.total_rows as u32,
            args.h_blocks,
            encoder,
        );
    }

    pub fn encode_dispatch_args(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassATileDispatchArguments<B>,
    ) {
        self.dispatch.encode(args.total_tiles, args.dispatch_args, args.num_tiles_y, encoder);
    }
}
