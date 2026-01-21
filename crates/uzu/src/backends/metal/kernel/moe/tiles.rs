use std::mem::size_of;
use std::ptr::NonNull;
use std::ffi::c_void;

use metal::MTLComputeCommandEncoder;

use crate::backends::metal::{
    BufferRef, CommandBufferRef, ComputePipelineState, MTLCommandBuffer,
    MTLCommandEncoder, MTLContext, MTLError, mtl_size,
};

// ---- Tile Kernels ----

#[derive(Debug, thiserror::Error)]
pub enum MoeTileError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

/// Arguments for tile counts encoder
pub struct MoeTileCountsArguments<'a> {
    pub offsets_buffer: BufferRef<'a>,     // [E+1]
    pub tile_counts_buffer: BufferRef<'a>, // [E]
    pub e: usize,
}

/// Arguments for tile scan encoder
pub struct MoeTileScanArguments<'a> {
    pub tile_counts_buffer: BufferRef<'a>, // [E]
    pub tile_offsets_buffer: BufferRef<'a>, // [E+1]
    pub total_tiles_buffer: BufferRef<'a>, // [>=2]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoeTileMapBuildArguments<'a> {
    pub expert_offsets: BufferRef<'a>, // [E+1]
    pub tile_offsets: BufferRef<'a>,   // [E+1]
    pub tile_counts: BufferRef<'a>,    // [E]
    pub tile_map: BufferRef<'a>,       // [total_tiles * 3]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoeTileDispatchArguments<'a> {
    pub total_tiles: BufferRef<'a>,   // [>=1]
    pub dispatch_args: BufferRef<'a>, // [3]
    pub num_tiles_x: u32,             // x dimension for indirect dispatch
}

pub struct MoeTileMapKernel {
    counts_pipeline: ComputePipelineState,
    scan_pipeline: ComputePipelineState,
    build_pipeline: ComputePipelineState,
    dispatch_pipeline: ComputePipelineState,
}

#[derive(Debug)]
pub struct MoePassATileCountsArguments<'a> {
    pub expert_offsets: BufferRef<'a>, // [E+1]
    pub tile_counts: BufferRef<'a>,    // [E]
    pub e: usize,
    pub h_blocks: u32,
}

#[derive(Debug)]
pub struct MoePassATileScanArguments<'a> {
    pub tile_counts: BufferRef<'a>,  // [E]
    pub tile_offsets: BufferRef<'a>, // [E+1]
    pub total_tiles: BufferRef<'a>,  // [>=1]
    pub e: usize,
}

#[derive(Debug)]
pub struct MoePassARowMapArguments<'a> {
    pub expert_offsets: BufferRef<'a>, // [E+1]
    pub row_expert_map: BufferRef<'a>, // [total_rows]
    pub total_rows: usize,
    pub e: usize,
}

#[derive(Debug)]
pub struct MoePassATileBuildArguments<'a> {
    pub expert_offsets: BufferRef<'a>, // [E+1]
    pub tile_offsets: BufferRef<'a>,   // [E+1]
    pub row_expert_map: BufferRef<'a>, // [total_rows]
    pub tile_map: BufferRef<'a>,       // [total_tiles * 3]
    pub total_rows: usize,
    pub h_blocks: u32,
}

#[derive(Debug)]
pub struct MoePassATileDispatchArguments<'a> {
    pub total_tiles: BufferRef<'a>,   // [>=1]
    pub dispatch_args: BufferRef<'a>, // [3]
    pub num_tiles_y: u32,
}

pub struct MoePassATileKernel {
    counts_pipeline: ComputePipelineState,
    scan_pipeline: ComputePipelineState,
    row_map_pipeline: ComputePipelineState,
    build_pipeline: ComputePipelineState,
    dispatch_pipeline: ComputePipelineState,
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
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.counts_pipeline);
        encoder.set_buffer(Some(args.expert_offsets), 0, 0);
        encoder.set_buffer(Some(args.tile_counts), 0, 1);
        let e_u32 = args.e as u32;
        unsafe {
            encoder.set_bytes(
                NonNull::new(&e_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                2,
            );
            encoder.set_bytes(
                NonNull::new(&args.h_blocks as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                3,
            );
        }
        encoder.dispatch_threadgroups(
            mtl_size(((args.e + 255) / 256) as u64, 1, 1),
            mtl_size(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scan(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassATileScanArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.scan_pipeline);
        encoder.set_buffer(Some(args.tile_counts), 0, 0);
        encoder.set_buffer(Some(args.tile_offsets), 0, 1);
        encoder.set_buffer(Some(args.total_tiles), 0, 2);
        let e_u32 = args.e as u32;
        unsafe {
            encoder.set_bytes(
                NonNull::new(&e_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                3,
            );
        }
        encoder.set_threadgroup_memory_length((1024 * size_of::<u32>()), 0);
        encoder.dispatch_threadgroups(mtl_size(1, 1, 1), mtl_size(1024, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_row_map(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassARowMapArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.row_map_pipeline);
        encoder.set_buffer(Some(args.expert_offsets), 0, 0);
        encoder.set_buffer(Some(args.row_expert_map), 0, 1);
        let total_rows_u32 = args.total_rows as u32;
        let e_u32 = args.e as u32;
        unsafe {
            encoder.set_bytes(
                NonNull::new(&total_rows_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                2,
            );
            encoder.set_bytes(
                NonNull::new(&e_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                3,
            );
        }
        encoder.dispatch_threadgroups(
            mtl_size(((args.total_rows + 255) / 256) as u64, 1, 1),
            mtl_size(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassATileBuildArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.build_pipeline);
        encoder.set_buffer(Some(args.expert_offsets), 0, 0);
        encoder.set_buffer(Some(args.tile_offsets), 0, 1);
        encoder.set_buffer(Some(args.row_expert_map), 0, 2);
        encoder.set_buffer(Some(args.tile_map), 0, 3);
        let total_rows_u32 = args.total_rows as u32;
        unsafe {
            encoder.set_bytes(
                NonNull::new(&total_rows_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                4,
            );
            encoder.set_bytes(
                NonNull::new(&args.h_blocks as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                5,
            );
        }
        let total_tiles_linear =
            (total_rows_u32 as u64).saturating_mul(args.h_blocks as u64);
        encoder.dispatch_threadgroups(
            mtl_size(((total_tiles_linear + 255) / 256) as u64, 1, 1),
            mtl_size(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoePassATileDispatchArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.dispatch_pipeline);
        encoder.set_buffer(Some(args.total_tiles), 0, 0);
        encoder.set_buffer(Some(args.dispatch_args), 0, 1);
        unsafe {
            encoder.set_bytes(
                NonNull::new(&args.num_tiles_y as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                2,
            );
        }
        encoder.dispatch_threadgroups(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
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
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.counts_pipeline);
        encoder.set_buffer(Some(args.offsets_buffer), 0, 0);
        encoder.set_buffer(Some(args.tile_counts_buffer), 0, 1);
        let e_u32 = args.e as u32;
        unsafe {
            encoder.set_bytes(
                NonNull::new(&e_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                2,
            );
        }
        encoder.dispatch_threadgroups(
            mtl_size(((args.e + 255) / 256) as u64, 1, 1),
            mtl_size(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scan(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoeTileScanArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.scan_pipeline);
        encoder.set_buffer(Some(args.tile_counts_buffer), 0, 0);
        encoder.set_buffer(Some(args.tile_offsets_buffer), 0, 1);
        encoder.set_buffer(Some(args.total_tiles_buffer), 0, 2);
        let e_u32 = args.e as u32;
        unsafe {
            encoder.set_bytes(
                NonNull::new(&e_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                3,
            );
        }
        encoder.dispatch_threadgroups(mtl_size(1, 1, 1), mtl_size(256, 1, 1));
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoeTileMapBuildArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.build_pipeline);
        encoder.set_buffer(Some(args.expert_offsets), 0, 0);
        encoder.set_buffer(Some(args.tile_offsets), 0, 1);
        encoder.set_buffer(Some(args.tile_counts), 0, 2);
        encoder.set_buffer(Some(args.tile_map), 0, 3);
        let e_u32 = args.e as u32;
        unsafe {
            encoder.set_bytes(
                NonNull::new(&e_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                4,
            );
        }
        encoder.dispatch_threadgroups(
            mtl_size(((args.e + 255) / 256) as u64, 1, 1),
            mtl_size(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &CommandBufferRef,
        args: &MoeTileDispatchArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.dispatch_pipeline);
        encoder.set_buffer(Some(args.total_tiles), 0, 0);
        encoder.set_buffer(Some(args.dispatch_args), 0, 1);
        unsafe {
            encoder.set_bytes(
                NonNull::new(&args.num_tiles_x as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                2,
            );
        }
        encoder.dispatch_threadgroups(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
        encoder.end_encoding();
        Ok(())
    }
}
