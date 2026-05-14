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
        offsets: &Allocation<B>,
        e: usize,
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[e], DataType::U32))?;
        self.counts.encode(offsets, &mut tile_counts, e as u32, encoder);
        Ok(tile_counts)
    }

    pub fn encode_scan(
        &self,
        encoder: &mut Encoder<B>,
        tile_counts: &Allocation<B>,
        e: usize,
    ) -> Result<MoeTileScanOutput<B>, B::Error> {
        let mut tile_offsets = encoder.allocate_scratch(size_for_shape(&[e + 1], DataType::U32))?;
        let mut total_tiles = encoder.allocate_scratch(size_for_shape(&[8], DataType::U32))?;
        self.scan.encode(tile_counts, &mut tile_offsets, &mut total_tiles, e as u32, encoder);
        Ok(MoeTileScanOutput {
            tile_offsets,
            total_tiles,
        })
    }

    pub fn encode_build_map(
        &self,
        encoder: &mut Encoder<B>,
        expert_offsets: &Allocation<B>,
        tile_offsets: &Allocation<B>,
        tile_counts: &Allocation<B>,
        total_rows: usize,
        e: usize,
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_map = encoder.allocate_scratch(size_for_shape(&[total_rows * 3], DataType::U32))?;
        self.build.encode(expert_offsets, tile_offsets, tile_counts, &mut tile_map, e as u32, encoder);
        Ok(tile_map)
    }

    pub fn encode_dispatch_args(
        &self,
        encoder: &mut Encoder<B>,
        total_tiles: &Allocation<B>,
        num_tiles_x: u32,
    ) -> Result<Allocation<B>, B::Error> {
        let mut dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(total_tiles, &mut dispatch_args, num_tiles_x, encoder);
        Ok(dispatch_args)
    }
}

pub struct MoeTileScanOutput<B: Backend> {
    pub tile_offsets: Allocation<B>,
    pub total_tiles: Allocation<B>,
}
