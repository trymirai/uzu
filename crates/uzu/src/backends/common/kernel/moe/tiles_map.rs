use crate::backends::common::{
    Backend, CommandBuffer, Kernels,
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
        command_buffer: &B::CommandBuffer,
        args: &MoeTileCountsArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.counts.encode(args.offsets_buffer, args.tile_counts_buffer, args.e as u32, encoder);
        });
    }

    pub fn encode_scan(
        &self,
        command_buffer: &B::CommandBuffer,
        args: &MoeTileScanArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.scan.encode(
                args.tile_counts_buffer,
                args.tile_offsets_buffer,
                args.total_tiles_buffer,
                args.e as u32,
                encoder,
            );
        });
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &B::CommandBuffer,
        args: &MoeTileMapBuildArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.build.encode(
                args.expert_offsets,
                args.tile_offsets,
                args.tile_counts,
                args.tile_map,
                args.e as u32,
                encoder,
            );
        });
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &B::CommandBuffer,
        args: &MoeTileDispatchArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| {
            self.dispatch.encode(args.total_tiles, args.dispatch_args, args.num_tiles_x, encoder);
        });
    }
}

pub struct MoeTileCountsArguments<'a, B: Backend> {
    pub offsets_buffer: &'a B::NativeBuffer,     // [E+1]
    pub tile_counts_buffer: &'a B::NativeBuffer, // [E]
    pub e: usize,
}

pub struct MoeTileScanArguments<'a, B: Backend> {
    pub tile_counts_buffer: &'a B::NativeBuffer,  // [E]
    pub tile_offsets_buffer: &'a B::NativeBuffer, // [E+1]
    pub total_tiles_buffer: &'a B::NativeBuffer,  // [>=2]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoeTileMapBuildArguments<'a, B: Backend> {
    pub expert_offsets: &'a B::NativeBuffer, // [E+1]
    pub tile_offsets: &'a B::NativeBuffer,   // [E+1]
    pub tile_counts: &'a B::NativeBuffer,    // [E]
    pub tile_map: &'a B::NativeBuffer,       // [total_tiles * 3]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoeTileDispatchArguments<'a, B: Backend> {
    pub total_tiles: &'a B::NativeBuffer,   // [>=1]
    pub dispatch_args: &'a B::NativeBuffer, // [3]
    pub num_tiles_x: u32,                   // x dimension for indirect dispatch
}
