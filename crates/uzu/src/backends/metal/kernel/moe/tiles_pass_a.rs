use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};
use objc2::{Message, runtime::ProtocolObject};

use crate::backends::{
    common::kernel::{
        MoePassABuildRowMapKernel, MoePassABuildTileMapKernel, MoePassATileCountsKernel, MoePassATileScanKernel,
        MoePassAWriteDispatchArgsKernel,
    },
    metal::{
        MetalContext, MetalError,
        kernel::dsl::{
            MoePassABuildRowMapMetalKernel, MoePassABuildTileMapMetalKernel, MoePassATileCountsMetalKernel,
            MoePassATileScanMetalKernel, MoePassAWriteDispatchArgsMetalKernel,
        },
    },
};

#[derive(Debug)]
pub struct MoePassATileCountsArguments<'a> {
    pub expert_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub tile_counts: &'a ProtocolObject<dyn MTLBuffer>,    // [E]
    pub e: usize,
    pub h_blocks: u32,
}

#[derive(Debug)]
pub struct MoePassATileScanArguments<'a> {
    pub tile_counts: &'a ProtocolObject<dyn MTLBuffer>,  // [E]
    pub tile_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub total_tiles: &'a ProtocolObject<dyn MTLBuffer>,  // [>=1]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoePassARowMapArguments<'a> {
    pub expert_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub row_expert_map: &'a ProtocolObject<dyn MTLBuffer>, // [total_rows]
    pub total_rows: usize,
    pub e: usize,
}

#[derive(Debug)]
pub struct MoePassATileBuildArguments<'a> {
    pub expert_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub tile_offsets: &'a ProtocolObject<dyn MTLBuffer>,   // [E+1]
    pub row_expert_map: &'a ProtocolObject<dyn MTLBuffer>, // [total_rows]
    pub tile_map: &'a ProtocolObject<dyn MTLBuffer>,       // [total_tiles * 3]
    pub total_rows: usize,
    pub h_blocks: u32,
}

#[derive(Debug)]
pub struct MoePassATileDispatchArguments<'a> {
    pub total_tiles: &'a ProtocolObject<dyn MTLBuffer>,   // [>=1]
    pub dispatch_args: &'a ProtocolObject<dyn MTLBuffer>, // [3]
    pub num_tiles_y: u32,
}

pub struct MoePassATileKernels {
    counts: MoePassATileCountsMetalKernel,
    scan: MoePassATileScanMetalKernel,
    row_map: MoePassABuildRowMapMetalKernel,
    build_map: MoePassABuildTileMapMetalKernel,
    dispatch: MoePassAWriteDispatchArgsMetalKernel,
}

impl MoePassATileKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            counts: MoePassATileCountsMetalKernel::new(ctx)?,
            scan: MoePassATileScanMetalKernel::new(ctx)?,
            row_map: MoePassABuildRowMapMetalKernel::new(ctx)?,
            build_map: MoePassABuildTileMapMetalKernel::new(ctx)?,
            dispatch: MoePassAWriteDispatchArgsMetalKernel::new(ctx)?,
        })
    }

    pub fn encode_counts(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoePassATileCountsArguments,
    ) {
        let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.counts.encode(
            &args.expert_offsets.retain(),
            &args.tile_counts.retain(),
            args.e as u32,
            args.h_blocks,
            &encoder,
        );
        encoder.end_encoding();
    }

    pub fn encode_scan(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoePassATileScanArguments,
    ) {
        let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.scan.encode(
            &args.tile_counts.retain(),
            &args.tile_offsets.retain(),
            &args.total_tiles.retain(),
            args.e as u32,
            &encoder,
        );
        encoder.end_encoding();
    }

    pub fn encode_row_map(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoePassARowMapArguments,
    ) {
        let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.row_map.encode(
            &args.expert_offsets.retain(),
            &args.row_expert_map.retain(),
            args.total_rows as u32,
            args.e as u32,
            &encoder,
        );
        encoder.end_encoding();
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoePassATileBuildArguments,
    ) {
        let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.build_map.encode(
            &args.expert_offsets.retain(),
            &args.tile_offsets.retain(),
            &args.row_expert_map.retain(),
            &args.tile_map.retain(),
            args.total_rows as u32,
            args.h_blocks,
            &encoder,
        );
        encoder.end_encoding();
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoePassATileDispatchArguments,
    ) {
        let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.dispatch.encode(&args.total_tiles.retain(), &args.dispatch_args.retain(), args.num_tiles_y, &encoder);
        encoder.end_encoding();
    }
}
