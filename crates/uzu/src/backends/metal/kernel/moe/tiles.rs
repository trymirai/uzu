use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{MTLContext, MTLError};

/// Arguments for tile counts encoder
pub struct MoeTileCountsArguments<'a> {
    pub offsets_buffer: &'a MTLBuffer,     // [E+1]
    pub tile_counts_buffer: &'a MTLBuffer, // [E]
    pub e: usize,
}

/// Encode MoE tile counts kernel (standalone, reusable)
pub fn encode_moe_tile_counts(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    args: &MoeTileCountsArguments,
) -> Result<(), MTLError> {
    MoeTileMapKernel::new(ctx)?
        .encode_counts(command_buffer, args)
        .map_err(Into::into)
}

/// Arguments for tile scan encoder
pub struct MoeTileScanArguments<'a> {
    pub tile_counts_buffer: &'a MTLBuffer, // [E]
    pub tile_offsets_buffer: &'a MTLBuffer, // [E+1]
    pub total_tiles_buffer: &'a MTLBuffer, // [>=2]
    pub e: usize,
}

/// Encode MoE tile scan kernel (standalone, reusable)
pub fn encode_moe_tile_scan(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    args: &MoeTileScanArguments,
) -> Result<(), MTLError> {
    MoeTileMapKernel::new(ctx)?
        .encode_scan(command_buffer, args)
        .map_err(Into::into)
}

// ---- Tile Kernels ----

#[derive(Debug, thiserror::Error)]
pub enum MoeTileError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

impl From<MoeTileError> for MTLError {
    fn from(err: MoeTileError) -> Self {
        match err {
            MoeTileError::MetalError(inner) => inner,
        }
    }
}

#[derive(Debug)]
pub struct MoeTileMapBuildArguments<'a> {
    pub expert_offsets: &'a MTLBuffer, // [E+1]
    pub tile_offsets: &'a MTLBuffer,   // [E+1]
    pub tile_counts: &'a MTLBuffer,    // [E]
    pub tile_map: &'a MTLBuffer,       // [total_tiles * 3]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoeTileDispatchArguments<'a> {
    pub total_tiles: &'a MTLBuffer,   // [>=1]
    pub dispatch_args: &'a MTLBuffer, // [3]
    pub num_tiles_x: u32,             // x dimension for indirect dispatch
}

pub struct MoeTileMapKernel {
    counts_pipeline: MTLComputePipelineState,
    scan_pipeline: MTLComputePipelineState,
    build_pipeline: MTLComputePipelineState,
    dispatch_pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoePassATileCountsArguments<'a> {
    pub expert_offsets: &'a MTLBuffer, // [E+1]
    pub tile_counts: &'a MTLBuffer,    // [E]
    pub e: usize,
    pub h_blocks: u32,
}

#[derive(Debug)]
pub struct MoePassATileScanArguments<'a> {
    pub tile_counts: &'a MTLBuffer,  // [E]
    pub tile_offsets: &'a MTLBuffer, // [E+1]
    pub total_tiles: &'a MTLBuffer,  // [>=1]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoePassARowMapArguments<'a> {
    pub expert_offsets: &'a MTLBuffer, // [E+1]
    pub row_expert_map: &'a MTLBuffer, // [total_rows]
    pub total_rows: usize,
    pub e: usize,
}

#[derive(Debug)]
pub struct MoePassATileBuildArguments<'a> {
    pub expert_offsets: &'a MTLBuffer, // [E+1]
    pub tile_offsets: &'a MTLBuffer,   // [E+1]
    pub row_expert_map: &'a MTLBuffer, // [total_rows]
    pub tile_map: &'a MTLBuffer,       // [total_tiles * 3]
    pub total_rows: usize,
    pub h_blocks: u32,
}

#[derive(Debug)]
pub struct MoePassATileDispatchArguments<'a> {
    pub total_tiles: &'a MTLBuffer,   // [>=1]
    pub dispatch_args: &'a MTLBuffer, // [3]
    pub num_tiles_y: u32,
}

pub struct MoePassATileKernel {
    counts_pipeline: MTLComputePipelineState,
    scan_pipeline: MTLComputePipelineState,
    row_map_pipeline: MTLComputePipelineState,
    build_pipeline: MTLComputePipelineState,
    dispatch_pipeline: MTLComputePipelineState,
}

impl MoePassATileKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeTileError> {
        Ok(Self {
            counts_pipeline: ctx
                .compute_pipeline_state("moe_pass_a_tile_counts", None)?,
            scan_pipeline: ctx
                .compute_pipeline_state("moe_pass_a_tile_scan", None)?,
            row_map_pipeline: ctx
                .compute_pipeline_state("moe_pass_a_build_row_map", None)?,
            build_pipeline: ctx
                .compute_pipeline_state("moe_pass_a_build_tile_map", None)?,
            dispatch_pipeline: ctx.compute_pipeline_state(
                "moe_pass_a_write_dispatch_args",
                None,
            )?,
        })
    }

    pub fn encode_counts(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassATileCountsArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.counts_pipeline);
        encoder.set_buffer(0, Some(args.expert_offsets), 0);
        encoder.set_buffer(1, Some(args.tile_counts), 0);
        let e_u32 = args.e as u32;
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &args.h_blocks as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scan(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassATileScanArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.scan_pipeline);
        encoder.set_buffer(0, Some(args.tile_counts), 0);
        encoder.set_buffer(1, Some(args.tile_offsets), 0);
        encoder.set_buffer(2, Some(args.total_tiles), 0);
        let e_u32 = args.e as u32;
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder
            .set_threadgroup_memory_length(0, (1024 * size_of::<u32>()) as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_row_map(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassARowMapArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.row_map_pipeline);
        encoder.set_buffer(0, Some(args.expert_offsets), 0);
        encoder.set_buffer(1, Some(args.row_expert_map), 0);
        let total_rows_u32 = args.total_rows as u32;
        let e_u32 = args.e as u32;
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &total_rows_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(((args.total_rows + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassATileBuildArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.build_pipeline);
        encoder.set_buffer(0, Some(args.expert_offsets), 0);
        encoder.set_buffer(1, Some(args.tile_offsets), 0);
        encoder.set_buffer(2, Some(args.row_expert_map), 0);
        encoder.set_buffer(3, Some(args.tile_map), 0);
        let total_rows_u32 = args.total_rows as u32;
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &total_rows_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &args.h_blocks as *const u32 as *const _,
        );
        let total_tiles_linear =
            (total_rows_u32 as u64).saturating_mul(args.h_blocks as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(((total_tiles_linear + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassATileDispatchArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.dispatch_pipeline);
        encoder.set_buffer(0, Some(args.total_tiles), 0);
        encoder.set_buffer(1, Some(args.dispatch_args), 0);
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &args.num_tiles_y as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }
}

impl MoeTileMapKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeTileError> {
        Ok(Self {
            counts_pipeline: ctx
                .compute_pipeline_state("moe_tile_counts", None)?,
            scan_pipeline: ctx.compute_pipeline_state("moe_tile_scan", None)?,
            build_pipeline: ctx
                .compute_pipeline_state("moe_build_tile_map", None)?,
            dispatch_pipeline: ctx
                .compute_pipeline_state("moe_write_dispatch_args", None)?,
        })
    }

    pub fn encode_counts(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoeTileCountsArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.counts_pipeline);
        encoder.set_buffer(0, Some(args.offsets_buffer), 0);
        encoder.set_buffer(1, Some(args.tile_counts_buffer), 0);
        let e_u32 = args.e as u32;
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scan(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoeTileScanArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.scan_pipeline);
        encoder.set_buffer(0, Some(args.tile_counts_buffer), 0);
        encoder.set_buffer(1, Some(args.tile_offsets_buffer), 0);
        encoder.set_buffer(2, Some(args.total_tiles_buffer), 0);
        let e_u32 = args.e as u32;
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoeTileMapBuildArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.build_pipeline);
        encoder.set_buffer(0, Some(args.expert_offsets), 0);
        encoder.set_buffer(1, Some(args.tile_offsets), 0);
        encoder.set_buffer(2, Some(args.tile_counts), 0);
        encoder.set_buffer(3, Some(args.tile_map), 0);
        let e_u32 = args.e as u32;
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoeTileDispatchArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.dispatch_pipeline);
        encoder.set_buffer(0, Some(args.total_tiles), 0);
        encoder.set_buffer(1, Some(args.dispatch_args), 0);
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &args.num_tiles_x as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }
}
