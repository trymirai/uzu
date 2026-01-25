use crate::backends::metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLContext, MTLError, MTLSize, ProtocolObject, Retained,
    metal_extensions::ComputeEncoderSetValue,
};

// ---- Tile Kernels ----

#[derive(Debug, thiserror::Error)]
pub enum MoeTileError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

/// Arguments for tile counts encoder
pub struct MoeTileCountsArguments<'a> {
    pub offsets_buffer: &'a ProtocolObject<dyn MTLBuffer>,     // [E+1]
    pub tile_counts_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [E]
    pub e: usize,
}

/// Arguments for tile scan encoder
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
    pub total_tiles: &'a ProtocolObject<dyn MTLBuffer>,   // [>=1]
    pub dispatch_args: &'a ProtocolObject<dyn MTLBuffer>, // [3]
    pub num_tiles_x: u32,             // x dimension for indirect dispatch
}

pub struct MoeTileMapKernel {
    counts_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scan_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    build_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dispatch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

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

pub struct MoePassATileKernel {
    counts_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scan_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    row_map_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    build_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dispatch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoePassATileCountsArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.counts_pipeline);
        encoder.set_buffer(Some(args.expert_offsets), 0, 0);
        encoder.set_buffer(Some(args.tile_counts), 0, 1);
        let e_u32 = args.e as u32;
        encoder.set_value(&e_u32, 2);
        encoder.set_value(&args.h_blocks, 3);
        encoder.dispatch_threadgroups(
            MTLSize::new((args.e + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scan(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
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
        encoder.set_value(&e_u32, 3);
        encoder.set_threadgroup_memory_length(1024 * size_of::<u32>() , 0);
        encoder.dispatch_threadgroups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_row_map(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
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
        encoder.set_value(&total_rows_u32, 2);
        encoder.set_value(&e_u32, 3);
        encoder.dispatch_threadgroups(
            MTLSize::new((args.total_rows + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
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
        encoder.set_value(&total_rows_u32, 4);
        encoder.set_value(&args.h_blocks, 5);
        let total_tiles_linear =
            (total_rows_u32 as u64).saturating_mul(args.h_blocks as u64);
        encoder.dispatch_threadgroups(
            MTLSize::new(((total_tiles_linear + 255) / 256) as usize, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoePassATileDispatchArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.dispatch_pipeline);
        encoder.set_buffer(Some(args.total_tiles), 0, 0);
        encoder.set_buffer(Some(args.dispatch_args), 0, 1);
        encoder.set_value(&args.num_tiles_y, 2);
        encoder.dispatch_threadgroups(
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
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoeTileCountsArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.counts_pipeline);
        encoder.set_buffer(Some(args.offsets_buffer), 0, 0);
        encoder.set_buffer(Some(args.tile_counts_buffer), 0, 1);
        let e_u32 = args.e as u32;
        encoder.set_value(&e_u32, 2);
        encoder.dispatch_threadgroups(
            MTLSize::new((args.e + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scan(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
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
        encoder.set_value(&e_u32, 3);
        encoder.dispatch_threadgroups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_build_map(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
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
        encoder.set_value(&e_u32, 4);
        encoder.dispatch_threadgroups(
            MTLSize::new((args.e + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub fn encode_dispatch_args(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: &MoeTileDispatchArguments,
    ) -> Result<(), MoeTileError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.dispatch_pipeline);
        encoder.set_buffer(Some(args.total_tiles), 0, 0);
        encoder.set_buffer(Some(args.dispatch_args), 0, 1);
        encoder.set_value(&args.num_tiles_x, 2);
        encoder.dispatch_threadgroups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1, 1, 1),
        );
        encoder.end_encoding();
        Ok(())
    }
}
