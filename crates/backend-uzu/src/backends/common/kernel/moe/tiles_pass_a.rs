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

pub struct MoePassATileCountsArguments<'a, B: Backend> {
    pub expert_offsets: &'a Allocation<B>, // [E+1]
    pub e: usize,
    pub h_blocks: u32,
}

pub struct MoePassATileScanArguments<'a, B: Backend> {
    pub tile_counts: &'a Allocation<B>, // [E]
    pub e: usize,
}

pub struct MoePassATileScanOutput<B: Backend> {
    pub tile_offsets: Allocation<B>,
    pub total_tiles: Allocation<B>,
}

pub struct MoePassARowMapArguments<'a, B: Backend> {
    pub expert_offsets: &'a Allocation<B>, // [E+1]
    pub total_rows: usize,
    pub e: usize,
}

pub struct MoePassATileBuildArguments<'a, B: Backend> {
    pub expert_offsets: &'a Allocation<B>, // [E+1]
    pub tile_offsets: &'a Allocation<B>,   // [E+1]
    pub row_expert_map: &'a Allocation<B>, // [total_rows]
    pub total_rows: usize,
    pub h_blocks: u32,
}

pub struct MoePassATileDispatchArguments<'a, B: Backend> {
    pub total_tiles: &'a Allocation<B>, // [>=1]
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
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_counts = encoder.allocate_scratch(size_for_shape(&[args.e], DataType::U32))?;
        self.counts.encode(args.expert_offsets, &mut tile_counts, args.e as u32, args.h_blocks, encoder);
        Ok(tile_counts)
    }

    pub fn encode_scan(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassATileScanArguments<B>,
    ) -> Result<MoePassATileScanOutput<B>, B::Error> {
        let mut tile_offsets = encoder.allocate_scratch(size_for_shape(&[args.e + 1], DataType::U32))?;
        let mut total_tiles = encoder.allocate_scratch(size_for_shape(&[1], DataType::U32))?;
        self.scan.encode(args.tile_counts, &mut tile_offsets, &mut total_tiles, args.e as u32, encoder);
        Ok(MoePassATileScanOutput {
            tile_offsets,
            total_tiles,
        })
    }

    pub fn encode_row_map(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassARowMapArguments<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut row_expert_map = encoder.allocate_scratch(size_for_shape(&[args.total_rows], DataType::U32))?;
        self.row_map.encode(args.expert_offsets, &mut row_expert_map, args.total_rows as u32, args.e as u32, encoder);
        Ok(row_expert_map)
    }

    pub fn encode_build_map(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassATileBuildArguments<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut tile_map =
            encoder.allocate_scratch(size_for_shape(&[args.total_rows * args.h_blocks as usize * 3], DataType::U32))?;
        self.build_map.encode(
            args.expert_offsets,
            args.tile_offsets,
            args.row_expert_map,
            &mut tile_map,
            args.total_rows as u32,
            args.h_blocks,
            encoder,
        );
        Ok(tile_map)
    }

    pub fn encode_dispatch_args(
        &self,
        encoder: &mut Encoder<B>,
        args: MoePassATileDispatchArguments<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut dispatch_args = encoder.allocate_scratch(size_for_shape(&[3], DataType::U32))?;
        self.dispatch.encode(args.total_tiles, &mut dispatch_args, args.num_tiles_y, encoder);
        Ok(dispatch_args)
    }
}
