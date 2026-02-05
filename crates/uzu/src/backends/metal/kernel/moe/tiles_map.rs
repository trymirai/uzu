use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};
use objc2::{__framework_prelude::ProtocolObject, Message};
use objc2::__framework_prelude::Retained;
use crate::backends::{
    common::kernel::{
        MoeBuildTileMapKernel, MoeTileCountsKernel, MoeTileScanKernel,
        MoeWriteDispatchArgsKernel,
    },
    metal::{
        MTLContext, MTLError,
        kernel::dsl::{
            MoeBuildTileMapMetalKernel, MoeTileCountsMetalKernel,
            MoeTileScanMetalKernel, MoeWriteDispatchArgsMetalKernel,
        },
    },
};

pub struct MoeTileMapKernels {
    counts: MoeTileCountsMetalKernel,
    scan: MoeTileScanMetalKernel,
    build: MoeBuildTileMapMetalKernel,
    dispatch: MoeWriteDispatchArgsMetalKernel,
}

impl MoeTileMapKernels {
    pub fn new(ctx: &MTLContext) -> Result<Self, MTLError> {
        Ok(Self {
            counts: MoeTileCountsMetalKernel::new(ctx)?,
            scan: MoeTileScanMetalKernel::new(ctx)?,
            build: MoeBuildTileMapMetalKernel::new(ctx)?,
            dispatch: MoeWriteDispatchArgsMetalKernel::new(ctx)?,
        })
    }

    pub fn encode_counts(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: &MoeTileCountsArguments,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.counts.encode(
            &args.offsets_buffer.retain(),
            &args.tile_counts_buffer.retain(),
            args.e as u32,
            &encoder,
        );
        encoder.end_encoding();
    }

    pub fn encode_scan(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: &MoeTileScanArguments,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.scan.encode(
            &args.tile_counts_buffer.retain(),
            &args.tile_offsets_buffer.retain(),
            &args.total_tiles_buffer.retain(),
            args.e as u32,
            &encoder,
        );
        encoder.end_encoding();
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: &MoeTileMapBuildArguments,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.build.encode(
            &args.expert_offsets.retain(),
            &args.tile_offsets.retain(),
            &args.tile_counts.retain(),
            &args.tile_map.retain(),
            args.e as u32,
            &encoder,
        );
        encoder.end_encoding();
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: &MoeTileDispatchArguments,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.dispatch.encode(
            &args.total_tiles.retain(),
            &args.dispatch_args.retain(),
            args.num_tiles_x,
            &encoder,
        );
        encoder.end_encoding();
    }
}

pub struct MoeTileCountsArguments<'a> {
    pub offsets_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub tile_counts_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [E]
    pub e: usize,
}

pub struct MoeTileScanArguments<'a> {
    pub tile_counts_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [E]
    pub tile_offsets_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub total_tiles_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [>=2]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoeTileMapBuildArguments<'a> {
    pub expert_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub tile_offsets: &'a ProtocolObject<dyn MTLBuffer>,   // [E+1]
    pub tile_counts: &'a ProtocolObject<dyn MTLBuffer>,    // [E]
    pub tile_map: &'a ProtocolObject<dyn MTLBuffer>,       // [total_tiles * 3]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoeTileDispatchArguments<'a> {
    pub total_tiles: &'a ProtocolObject<dyn MTLBuffer>, // [>=1]
    pub dispatch_args: &'a ProtocolObject<dyn MTLBuffer>, // [3]
    pub num_tiles_x: u32, // x dimension for indirect dispatch
}
