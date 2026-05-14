use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            Kernels, MoePassABuildRowMapKernel, MoePassABuildTileMapKernel, MoePassATileCountsKernel,
            MoePassATileScanKernel, MoePassAWriteDispatchArgsKernel,
        },
    },
};

pub struct MoePassATileScanOutput<B: Backend> {
    pub tile_offsets: Allocation<B>,
    pub total_tiles: Allocation<B>,
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
        expert_offsets: &Allocation<B>,
        e: usize,
        h_blocks: u32,
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[e], DataType::U32))?;
        self.counts.encode(expert_offsets, &mut tile_counts, e as u32, h_blocks, encoder);
        Ok(tile_counts)
    }

    pub fn encode_scan(
        &self,
        encoder: &mut Encoder<B>,
        tile_counts: &Allocation<B>,
        e: usize,
    ) -> Result<MoePassATileScanOutput<B>, B::Error> {
        let mut tile_offsets = encoder.allocate_scratch(size_for_shape(&[e + 1], DataType::U32))?;
        let mut total_tiles = encoder.allocate_scratch(size_for_shape(&[1], DataType::U32))?;
        self.scan.encode(tile_counts, &mut tile_offsets, &mut total_tiles, e as u32, encoder);
        Ok(MoePassATileScanOutput {
            tile_offsets,
            total_tiles,
        })
    }

    pub fn encode_row_map(
        &self,
        encoder: &mut Encoder<B>,
        expert_offsets: &Allocation<B>,
        total_rows: usize,
        e: usize,
    ) -> Result<Allocation<B>, B::Error> {
        let mut row_expert_map = encoder.allocate_scratch(size_for_shape(&[total_rows], DataType::U32))?;
        self.row_map.encode(expert_offsets, &mut row_expert_map, total_rows as u32, e as u32, encoder);
        Ok(row_expert_map)
    }

    pub fn encode_build_map(
        &self,
        encoder: &mut Encoder<B>,
        expert_offsets: &Allocation<B>,
        tile_offsets: &Allocation<B>,
        row_expert_map: &Allocation<B>,
        total_rows: usize,
        h_blocks: u32,
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_map =
            encoder.allocate_scratch(size_for_shape(&[total_rows * h_blocks as usize * 3], DataType::U32))?;
        self.build_map.encode(
            expert_offsets,
            tile_offsets,
            row_expert_map,
            &mut tile_map,
            total_rows as u32,
            h_blocks,
            encoder,
        );
        Ok(tile_map)
    }

    pub fn encode_dispatch_args(
        &self,
        encoder: &mut Encoder<B>,
        total_tiles: &Allocation<B>,
        num_tiles_y: u32,
    ) -> Result<Allocation<B>, B::Error> {
        let mut dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(total_tiles, &mut dispatch_args, num_tiles_y, encoder);
        Ok(dispatch_args)
    }
}
