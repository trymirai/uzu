use crate::backends::common::{
    CommandBuffer,
    kernel::{
        Backend, Kernels, MoePassABuildRowMapKernel, MoePassABuildTileMapKernel, MoePassATileCountsKernel,
        MoePassATileScanKernel, MoePassAWriteDispatchArgsKernel,
    },
};

#[derive(Debug)]
pub struct MoePassATileCountsArguments<'a, B: Backend> {
    pub expert_offsets: &'a B::NativeBuffer, // [E+1]
    pub tile_counts: &'a B::NativeBuffer,    // [E]
    pub e: usize,
    pub h_blocks: u32,
}

#[derive(Debug)]
pub struct MoePassATileScanArguments<'a, B: Backend> {
    pub tile_counts: &'a B::NativeBuffer,  // [E]
    pub tile_offsets: &'a B::NativeBuffer, // [E+1]
    pub total_tiles: &'a B::NativeBuffer,  // [>=1]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoePassARowMapArguments<'a, B: Backend> {
    pub expert_offsets: &'a B::NativeBuffer, // [E+1]
    pub row_expert_map: &'a B::NativeBuffer, // [total_rows]
    pub total_rows: usize,
    pub e: usize,
}

#[derive(Debug)]
pub struct MoePassATileBuildArguments<'a, B: Backend> {
    pub expert_offsets: &'a B::NativeBuffer, // [E+1]
    pub tile_offsets: &'a B::NativeBuffer,   // [E+1]
    pub row_expert_map: &'a B::NativeBuffer, // [total_rows]
    pub tile_map: &'a B::NativeBuffer,       // [total_tiles * 3]
    pub total_rows: usize,
    pub h_blocks: u32,
}

#[derive(Debug)]
pub struct MoePassATileDispatchArguments<'a, B: Backend> {
    pub total_tiles: &'a B::NativeBuffer,   // [>=1]
    pub dispatch_args: &'a B::NativeBuffer, // [3]
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
        command_buffer: &B::CommandBuffer,
        args: &MoePassATileCountsArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.counts.encode(args.expert_offsets, args.tile_counts, args.e as u32, args.h_blocks, encoder);
        });
    }

    pub fn encode_scan(
        &self,
        command_buffer: &B::CommandBuffer,
        args: &MoePassATileScanArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.scan.encode(args.tile_counts, args.tile_offsets, args.total_tiles, args.e as u32, encoder);
        });
    }

    pub fn encode_row_map(
        &self,
        command_buffer: &B::CommandBuffer,
        args: &MoePassARowMapArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.row_map.encode(
                args.expert_offsets,
                args.row_expert_map,
                args.total_rows as u32,
                args.e as u32,
                encoder,
            );
        });
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &B::CommandBuffer,
        args: &MoePassATileBuildArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.build_map.encode(
                args.expert_offsets,
                args.tile_offsets,
                args.row_expert_map,
                args.tile_map,
                args.total_rows as u32,
                args.h_blocks,
                encoder,
            );
        });
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &B::CommandBuffer,
        args: &MoePassATileDispatchArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.dispatch.encode(args.total_tiles, args.dispatch_args, args.num_tiles_y, encoder);
        });
    }
}
