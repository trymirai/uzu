use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{MoeBuildTileMapKernel, MoeTileCountsKernel, MoeTileScanKernel, MoeWriteDispatchArgsKernel},
    },
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
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[args.e], DataType::U32))?;
        self.counts.encode(args.offsets, &mut tile_counts, args.e as u32, encoder);
        Ok(tile_counts)
    }

    pub fn encode_scan(
        &self,
        encoder: &mut Encoder<B>,
        args: MoeTileScanArguments<B>,
    ) -> Result<MoeTileScanOutput<B>, B::Error> {
        let mut tile_offsets = encoder.allocate_scratch(size_for_shape(&[args.e + 1], DataType::U32))?;
        let mut total_tiles = encoder.allocate_scratch(size_for_shape(&[8], DataType::U32))?;
        self.scan.encode(args.tile_counts, &mut tile_offsets, &mut total_tiles, args.e as u32, encoder);
        Ok(MoeTileScanOutput {
            tile_offsets,
            total_tiles,
        })
    }

    pub fn encode_build_map(
        &self,
        encoder: &mut Encoder<B>,
        args: MoeTileMapBuildArguments<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_map = encoder.allocate_scratch(size_for_shape(&[args.total_rows * 3], DataType::U32))?;
        self.build.encode(
            args.expert_offsets,
            args.tile_offsets,
            args.tile_counts,
            &mut tile_map,
            args.e as u32,
            encoder,
        );
        Ok(tile_map)
    }

    pub fn encode_dispatch_args(
        &self,
        encoder: &mut Encoder<B>,
        args: MoeTileDispatchArguments<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(args.total_tiles, &mut dispatch_args, args.num_tiles_x, encoder);
        Ok(dispatch_args)
    }
}

pub struct MoeTileCountsArguments<'a, B: Backend> {
    pub offsets: &'a Allocation<B>, // [E+1]
    pub e: usize,
}

pub struct MoeTileScanArguments<'a, B: Backend> {
    pub tile_counts: &'a Allocation<B>, // [E]
    pub e: usize,
}

pub struct MoeTileScanOutput<B: Backend> {
    pub tile_offsets: Allocation<B>,
    pub total_tiles: Allocation<B>,
}

pub struct MoeTileMapBuildArguments<'a, B: Backend> {
    pub expert_offsets: &'a Allocation<B>, // [E+1]
    pub tile_offsets: &'a Allocation<B>,   // [E+1]
    pub tile_counts: &'a Allocation<B>,    // [E]
    pub total_rows: usize,
    pub e: usize,
}

pub struct MoeTileDispatchArguments<'a, B: Backend> {
    pub total_tiles: &'a Allocation<B>, // [>=1]
    pub num_tiles_x: u32,               // x dimension for indirect dispatch
}
