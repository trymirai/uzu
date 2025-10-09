use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, FunctionConstantValues,
    MTLDataType, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

mod encodable;
pub use encodable::{MoeBlockEncodable, SharedMoeWeights};

fn dtype_suffix(dtype: KernelDataType) -> &'static str {
    match dtype {
        KernelDataType::Float16 => "f16",
        KernelDataType::BFloat16 => "bf16",
        KernelDataType::Float32 => "f32",
    }
}

fn dtype_index(dtype: KernelDataType) -> usize {
    match dtype {
        KernelDataType::Float16 => 0,
        KernelDataType::BFloat16 => 1,
        KernelDataType::Float32 => 2,
    }
}

// ---- Router Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeRouterError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeRouterKernel {
    pipeline_f16: MTLComputePipelineState,
    pipeline_f32: MTLComputePipelineState,
    pipeline_bf16: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeRouterArguments<'a> {
    pub input_buffer: &'a MTLBuffer,  // [T, d_model]
    pub weight_buffer: &'a MTLBuffer, // [E, d_model]
    pub bias_buffer: &'a MTLBuffer,   // [E]
    pub output_buffer: &'a MTLBuffer, // [T, E]
    pub t: usize,
    pub d_model: usize,
    pub e: usize,
}

// Legacy alias for backward compatibility
pub type RouterEncoderArgs<'a> = MoeRouterArguments<'a>;

/// Encode MoE router kernel (standalone, reusable)
pub fn encode_moe_router(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    dtype: KernelDataType,
    args: RouterEncoderArgs,
) -> Result<(), MTLError> {
    let kernel_name = match dtype {
        KernelDataType::BFloat16 => "moe_router_bf16",
        KernelDataType::Float16 => "moe_router_f16",
        KernelDataType::Float32 => "moe_router_f32",
    };

    let pipeline = ctx.compute_pipeline_state(kernel_name, None)?;
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(args.input_buffer), 0);
    encoder.set_buffer(1, Some(args.weight_buffer), 0);
    encoder.set_buffer(2, Some(args.bias_buffer), 0);
    encoder.set_buffer(3, Some(args.output_buffer), 0);

    let t_u32 = args.t as u32;
    let d_u32 = args.d_model as u32;
    let e_u32 = args.e as u32;
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        &d_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );

    // Optimized: 8 simdgroups per TG (256 threads) with TG input caching
    let num_simdgroups: u64 = 8;
    let tg_x = (args.e as u64 + num_simdgroups - 1) / num_simdgroups;
    encoder.dispatch_thread_groups(
        MTLSize::new(tg_x, args.t as u64, 1),
        MTLSize::new(32 * num_simdgroups, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

/// Encode MoE router kernel using a prebuilt pipeline (avoids pipeline lookup)
pub fn encode_moe_router_with_pipeline(
    pipeline: &MTLComputePipelineState,
    command_buffer: &CommandBufferRef,
    args: RouterEncoderArgs,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(args.input_buffer), 0);
    encoder.set_buffer(1, Some(args.weight_buffer), 0);
    encoder.set_buffer(2, Some(args.bias_buffer), 0);
    encoder.set_buffer(3, Some(args.output_buffer), 0);

    let t_u32 = args.t as u32;
    let d_u32 = args.d_model as u32;
    let e_u32 = args.e as u32;
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        &d_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );

    // Optimized: 8 simdgroups per TG (256 threads) with TG input caching
    let num_simdgroups: u64 = 8;
    let tg_x = (args.e as u64 + num_simdgroups - 1) / num_simdgroups;
    encoder.dispatch_thread_groups(
        MTLSize::new(tg_x, args.t as u64, 1),
        MTLSize::new(32 * num_simdgroups, 1, 1),
    );
    encoder.end_encoding();
}

impl MoeRouterKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeRouterError> {
        let pipeline_f16 =
            mtl_context.compute_pipeline_state("moe_router_f16", None)?;
        let pipeline_f32 =
            mtl_context.compute_pipeline_state("moe_router_f32", None)?;
        let pipeline_bf16 =
            mtl_context.compute_pipeline_state("moe_router_bf16", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_f32,
            pipeline_bf16,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        dtype: KernelDataType,
        args: MoeRouterArguments,
    ) -> Result<(), MoeRouterError> {
        let compute_encoder = command_buffer.new_compute_command_encoder();
        match dtype {
            KernelDataType::Float16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f16)
            },
            KernelDataType::Float32 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f32)
            },
            KernelDataType::BFloat16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_bf16)
            },
        }

        compute_encoder.set_buffer(0, Some(args.input_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.weight_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.bias_buffer), 0);
        compute_encoder.set_buffer(3, Some(args.output_buffer), 0);

        let t_u32 = args.t as u32;
        let d_u32 = args.d_model as u32;
        let e_u32 = args.e as u32;
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_u32 as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );

        // Optimized: 8 simdgroups per TG (256 threads) with TG input caching
        let num_simdgroups: u64 = 8;
        let tg_x = (args.e as u64 + num_simdgroups - 1) / num_simdgroups;
        compute_encoder.dispatch_thread_groups(
            MTLSize::new(tg_x, args.t as u64, 1),
            MTLSize::new(32 * num_simdgroups, 1, 1),
        );
        compute_encoder.end_encoding();
        Ok(())
    }
}

/// Encode MoE TopK kernel (standalone, reusable)
pub fn encode_moe_topk(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    dtype: KernelDataType,
    args: &MoeTopKArguments,
) -> Result<(), MTLError> {
    let kernel_name = match dtype {
        KernelDataType::BFloat16 => "moe_topk_select_bf16",
        KernelDataType::Float16 => "moe_topk_select_f16",
        KernelDataType::Float32 => "moe_topk_select_f32",
    };

    let pipeline = ctx.compute_pipeline_state(kernel_name, None)?;
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(args.logits_buffer), 0);
    encoder.set_buffer(1, Some(args.topk_ids_buffer), 0);
    encoder.set_buffer(2, Some(args.topk_probs_buffer), 0);

    let t_u32 = args.t as u32;
    let e_u32 = args.e as u32;
    let k_u32 = args.k as u32;
    let renorm_u32 = if args.renorm {
        1u32
    } else {
        0u32
    };
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        &renorm_u32 as *const u32 as *const _,
    );

    encoder.dispatch_thread_groups(
        MTLSize::new(args.t as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

/// Encode MoE TopK using a prebuilt pipeline (avoids pipeline lookup)
pub fn encode_moe_topk_with_pipeline(
    pipeline: &MTLComputePipelineState,
    command_buffer: &CommandBufferRef,
    args: &MoeTopKArguments,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(args.logits_buffer), 0);
    encoder.set_buffer(1, Some(args.topk_ids_buffer), 0);
    encoder.set_buffer(2, Some(args.topk_probs_buffer), 0);

    let t_u32 = args.t as u32;
    let e_u32 = args.e as u32;
    let k_u32 = args.k as u32;
    let renorm_u32 = if args.renorm {
        1u32
    } else {
        0u32
    };
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        &renorm_u32 as *const u32 as *const _,
    );

    encoder.dispatch_thread_groups(
        MTLSize::new(args.t as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
}

/// Encode MoE bucket counts kernel (standalone, reusable)
pub fn encode_moe_bucket_counts(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    args: &MoeBucketCountsArguments,
) -> Result<(), MTLError> {
    let partials_pipeline =
        ctx.compute_pipeline_state("moe_bucket_partials", None)?;
    let reduce_pipeline =
        ctx.compute_pipeline_state("moe_bucket_reduce_partials", None)?;

    // Stage 1: partials
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&partials_pipeline);
    encoder.set_buffer(0, Some(args.topk_ids_buffer), 0);
    encoder.set_buffer(1, Some(args.partials_buffer), 0);
    let t_u32 = args.t as u32;
    let e_u32 = args.e as u32;
    let k_u32 = args.k as u32;
    encoder.set_bytes(
        2,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const _,
    );

    let num_blocks = ((args.t + 255) / 256).max(1);
    let num_tiles = ((args.e + 512 - 1) / 512).max(1);
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tiles as u64, num_blocks as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();

    // Stage 2: reduce
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&reduce_pipeline);
    encoder.set_buffer(0, Some(args.partials_buffer), 0);
    encoder.set_buffer(1, Some(args.counts_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    let num_blocks_u32 = num_blocks as u32;
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &num_blocks_u32 as *const u32 as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

/// Encode MoE bucket counts using prebuilt pipelines
pub fn encode_moe_bucket_counts_with_pipelines(
    partials_pipeline: &MTLComputePipelineState,
    reduce_pipeline: &MTLComputePipelineState,
    command_buffer: &CommandBufferRef,
    args: &MoeBucketCountsArguments,
) {
    // Stage 1: partials
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(partials_pipeline);
    encoder.set_buffer(0, Some(args.topk_ids_buffer), 0);
    encoder.set_buffer(1, Some(args.partials_buffer), 0);
    let t_u32 = args.t as u32;
    let e_u32 = args.e as u32;
    let k_u32 = args.k as u32;
    encoder.set_bytes(
        2,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const _,
    );
    let num_blocks = ((args.t + 255) / 256).max(1);
    let num_tiles = ((args.e + 512 - 1) / 512).max(1);
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tiles as u64, num_blocks as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();

    // Stage 2: reduce
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(reduce_pipeline);
    encoder.set_buffer(0, Some(args.partials_buffer), 0);
    encoder.set_buffer(1, Some(args.counts_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    let num_blocks_u32 = num_blocks as u32;
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &num_blocks_u32 as *const u32 as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
}

/// Encode MoE offsets scan kernel (standalone, reusable)
pub fn encode_moe_offsets_scan(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    args: &MoeOffsetsScanArguments,
) -> Result<(), MTLError> {
    let pipeline =
        ctx.compute_pipeline_state("moe_offsets_exclusive_scan", None)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(args.counts_buffer), 0);
    encoder.set_buffer(1, Some(args.offsets_buffer), 0);
    encoder.set_buffer(2, Some(args.sumk_buffer), 0);
    let e_u32 = args.e as u32;
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    encoder
        .dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    Ok(())
}

pub fn encode_moe_offsets_scan_with_pipeline(
    pipeline: &MTLComputePipelineState,
    command_buffer: &CommandBufferRef,
    args: &MoeOffsetsScanArguments,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(args.counts_buffer), 0);
    encoder.set_buffer(1, Some(args.offsets_buffer), 0);
    encoder.set_buffer(2, Some(args.sumk_buffer), 0);
    let e_u32 = args.e as u32;
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    encoder
        .dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
}

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

/// Encode MoE gather kernel (standalone, reusable)
pub fn encode_moe_gather(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    dtype: KernelDataType,
    args: &MoeGatherArguments,
) -> Result<(), MTLError> {
    // Select kernel based on d_model size
    const USE_2D_THRESHOLD: usize = 512;
    const BF16_ROWS_PER_TG: usize = 8;
    let (kernel_name, use_2d) = if args.d_model >= USE_2D_THRESHOLD {
        match dtype {
            KernelDataType::BFloat16 => ("moe_gather_x_perm_bf16_2d", true),
            KernelDataType::Float16 => ("moe_gather_x_perm_f16_2d", true),
            KernelDataType::Float32 => ("moe_gather_x_perm_f32_2d", true),
        }
    } else {
        match dtype {
            KernelDataType::BFloat16 => ("moe_gather_x_perm_bf16", false),
            KernelDataType::Float16 => ("moe_gather_x_perm_f16", false),
            KernelDataType::Float32 => ("moe_gather_x_perm_f32", false),
        }
    };

    let pipeline = ctx.compute_pipeline_state(kernel_name, None)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(args.x_buffer), 0);
    encoder.set_buffer(1, Some(args.bucketed_ids_buffer), 0);
    encoder.set_buffer(2, Some(args.x_perm_buffer), 0);
    encoder.set_buffer(3, Some(args.sumk_buffer), 0);
    let d_model_u32 = args.d_model as u32;
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &d_model_u32 as *const u32 as *const _,
    );

    let max_rows = args.t * args.k;
    if max_rows > 0 {
        let tg_x = if use_2d {
            (max_rows + BF16_ROWS_PER_TG - 1) / BF16_ROWS_PER_TG
        } else {
            (max_rows + 255) / 256
        };
        encoder.dispatch_thread_groups(
            MTLSize::new(tg_x as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    encoder.end_encoding();
    Ok(())
}

pub fn encode_moe_gather_with_pipeline(
    pipeline: &MTLComputePipelineState,
    command_buffer: &CommandBufferRef,
    args: &MoeGatherArguments,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(args.x_buffer), 0);
    encoder.set_buffer(1, Some(args.bucketed_ids_buffer), 0);
    encoder.set_buffer(2, Some(args.x_perm_buffer), 0);
    encoder.set_buffer(3, Some(args.sumk_buffer), 0);
    let d_model_u32 = args.d_model as u32;
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &d_model_u32 as *const u32 as *const _,
    );
    let max_rows = args.t * args.k;
    if max_rows > 0 {
        // Choose dispatch based on d_model: 2D tiled for large d_model, 1D for small
        const USE_2D_THRESHOLD: usize = 512;
        if args.d_model >= USE_2D_THRESHOLD {
            // 2D tiled dispatch: 8 rows per threadgroup
            const ROWS_PER_TG: u64 = 8;
            let num_tgs = (max_rows as u64 + ROWS_PER_TG - 1) / ROWS_PER_TG;
            encoder.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        } else {
            // 1D dispatch (baseline)
            encoder.dispatch_thread_groups(
                MTLSize::new(((max_rows + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
    }
    encoder.end_encoding();
}

/// Encode MoE finalize kernel (standalone, reusable)
pub fn encode_moe_finalize(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    dtype: KernelDataType,
    args: &MoeFinalizeArguments,
) -> Result<(), MTLError> {
    let kernel_name = match dtype {
        KernelDataType::BFloat16 => "moe_finalize_bf16",
        KernelDataType::Float16 => "moe_finalize_f16",
        KernelDataType::Float32 => "moe_finalize_f32",
    };

    let pipeline = ctx.compute_pipeline_state(kernel_name, None)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(args.tok2row_buffer), 0);
    encoder.set_buffer(1, Some(args.probs_buffer), 0);
    encoder.set_buffer(2, Some(args.y_partial_buffer), 0);
    encoder.set_buffer(3, Some(args.y_out_buffer), 0);
    let t_u32 = args.t as u32;
    let d_model_u32 = args.d_model as u32;
    let k_u32 = args.k as u32;
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        &d_model_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const _,
    );

    encoder.dispatch_thread_groups(
        MTLSize::new(args.t as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

pub fn encode_moe_finalize_with_pipeline(
    pipeline: &MTLComputePipelineState,
    command_buffer: &CommandBufferRef,
    args: &MoeFinalizeArguments,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(args.tok2row_buffer), 0);
    encoder.set_buffer(1, Some(args.probs_buffer), 0);
    encoder.set_buffer(2, Some(args.y_partial_buffer), 0);
    encoder.set_buffer(3, Some(args.y_out_buffer), 0);
    let t_u32 = args.t as u32;
    let d_model_u32 = args.d_model as u32;
    let k_u32 = args.k as u32;
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        &d_model_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const _,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(args.t as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
}

#[derive(Debug, thiserror::Error)]
pub enum MoeTopKError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Invalid dimensions: T={t}, E={e}, K={k}")]
    InvalidDimensions {
        t: usize,
        e: usize,
        k: usize,
    },
}

pub struct MoeTopKKernel {
    pipeline_f16: MTLComputePipelineState,
    pipeline_f32: MTLComputePipelineState,
    pipeline_bf16: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeTopKArguments<'a> {
    pub logits_buffer: &'a MTLBuffer,
    pub topk_ids_buffer: &'a MTLBuffer,
    pub topk_probs_buffer: &'a MTLBuffer,
    pub t: usize,
    pub e: usize,
    pub k: usize,
    pub renorm: bool,
}

// ---- Bucket Counts Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeBucketCountsError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Invalid dimensions: T={t}, E={e}, K={k}")]
    InvalidDimensions {
        t: usize,
        e: usize,
        k: usize,
    },
}

pub struct MoeBucketCountsKernel {
    pipeline_partials: MTLComputePipelineState,
    pipeline_reduce: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeBucketCountsArguments<'a> {
    pub topk_ids_buffer: &'a MTLBuffer,
    pub counts_buffer: &'a MTLBuffer,
    pub partials_buffer: &'a MTLBuffer,
    pub t: usize,
    pub e: usize,
    pub k: usize,
}

// ---- Offsets Scan Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeOffsetsScanError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeOffsetsScanKernel {
    pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeOffsetsScanArguments<'a> {
    pub counts_buffer: &'a MTLBuffer,  // [E]
    pub offsets_buffer: &'a MTLBuffer, // [E+1]
    pub sumk_buffer: &'a MTLBuffer,    // [1]
    pub e: usize,
}

impl MoeOffsetsScanKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeOffsetsScanError> {
        let pipeline = mtl_context
            .compute_pipeline_state("moe_offsets_exclusive_scan", None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeOffsetsScanArguments,
    ) -> Result<(), MoeOffsetsScanError> {
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(0, Some(args.counts_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.offsets_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.sumk_buffer), 0);
        let e_u32 = args.e as u32;
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );

        // Single threadgroup implementation (BLOCK_SIZE=256), repeated in-kernel
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);
        compute_encoder
            .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        compute_encoder.end_encoding();
        Ok(())
    }
}

// ---- Gather Permuted Activations Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeGatherError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeGatherKernel {
    pipeline_f16: MTLComputePipelineState,
    pipeline_f32: MTLComputePipelineState,
    pipeline_bf16: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeGatherArguments<'a> {
    pub x_buffer: &'a MTLBuffer,
    pub bucketed_ids_buffer: &'a MTLBuffer,
    pub x_perm_buffer: &'a MTLBuffer,
    pub sumk_buffer: &'a MTLBuffer,
    pub t: usize,
    pub k: usize,
    pub d_model: usize,
}

impl MoeGatherKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeGatherError> {
        let pipeline_f16 =
            ctx.compute_pipeline_state("moe_gather_x_perm_f16", None)?;
        let pipeline_f32 =
            ctx.compute_pipeline_state("moe_gather_x_perm_f32", None)?;
        let pipeline_bf16 =
            ctx.compute_pipeline_state("moe_gather_x_perm_bf16_2d", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_f32,
            pipeline_bf16,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        dtype: KernelDataType,
        args: MoeGatherArguments,
    ) -> Result<(), MoeGatherError> {
        let encoder = command_buffer.new_compute_command_encoder();
        match dtype {
            KernelDataType::Float16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_f16);
            },
            KernelDataType::Float32 => {
                encoder.set_compute_pipeline_state(&self.pipeline_f32);
            },
            KernelDataType::BFloat16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_bf16);
            },
        }

        encoder.set_buffer(0, Some(args.x_buffer), 0);
        encoder.set_buffer(1, Some(args.bucketed_ids_buffer), 0);
        encoder.set_buffer(2, Some(args.x_perm_buffer), 0);
        encoder.set_buffer(3, Some(args.sumk_buffer), 0);
        let d_model_u32 = args.d_model as u32;
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );

        let max_rows = args.t * args.k;
        if max_rows == 0 {
            encoder.end_encoding();
            return Ok(());
        }
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = match dtype {
            KernelDataType::BFloat16 => {
                const BF16_ROWS_PER_TG: usize = 8;
                MTLSize::new(
                    ((max_rows + BF16_ROWS_PER_TG - 1) / BF16_ROWS_PER_TG)
                        as u64,
                    1,
                    1,
                )
            },
            _ => MTLSize::new(((max_rows + 255) / 256) as u64, 1, 1),
        };
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        Ok(())
    }
}

// ---- Scatter Buckets Kernels ----

#[derive(Debug, thiserror::Error)]
pub enum MoeScatterError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}
#[derive(Debug, thiserror::Error)]
pub enum MoeExpertsError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("{0}")]
    Generic(String),
}

impl From<MoeTileError> for MoeExpertsError {
    fn from(err: MoeTileError) -> Self {
        match err {
            MoeTileError::MetalError(inner) => {
                MoeExpertsError::MetalError(inner)
            },
        }
    }
}

pub struct MoeExpertsFusedKernel {
    pipelines: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype]
    tile_map: MoeTileMapKernel,
}

#[derive(Debug)]
pub struct MoeExpertsArguments<'a> {
    pub x_perm_buffer: &'a MTLBuffer, // [sum_k, d_model] - permuted input
    pub expert_offsets: &'a MTLBuffer, // [E+1] - expert segment offsets
    pub w13_all: &'a MTLBuffer, // [E, 2*d_ff, d_model] - transposed up projection weights
    pub w2_all: &'a MTLBuffer, // [E, d_model, d_ff] - transposed down projection weights
    pub y_partial: &'a MTLBuffer, // [sum_k, d_model] - output buffer
    pub up_biases: &'a MTLBuffer, // [E, 2*d_ff] - up projection biases
    pub down_biases: &'a MTLBuffer, // [E, d_model] - down projection biases
    pub tile_counts: &'a MTLBuffer, // [E]
    pub tile_row_offsets: &'a MTLBuffer, // [E+1]
    pub tile_map: &'a MTLBuffer, // [max_tiles * 3]
    pub total_tiles: &'a MTLBuffer, // [2]
    pub dispatch_args: &'a MTLBuffer, // [3]
    pub num_tiles_n: usize,
    pub t: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub e: usize,
    pub k: usize,
    pub gating_code: u32,
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
    pub data_type: KernelDataType,
}

pub struct MoeExpertsTwoPassDecodeKernel {
    pass_a_tile: MoePassATileKernel,
    pass_a_indirect: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype]
    fused_down: Vec<MTLComputePipelineState>,           // [dtype]
}

pub struct MoeExpertsTwoPassPrefillKernel {
    tile_map: MoeTileMapKernel,
    pass_a_indirect: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype]
    pass_b_indirect: Vec<MTLComputePipelineState>,      // [dtype]
}

#[derive(Debug)]
pub struct MoeExpertsTwoPassArguments<'a> {
    pub x_perm_buffer: &'a MTLBuffer,
    pub expert_offsets: &'a MTLBuffer,
    pub row_expert_map: &'a MTLBuffer,
    pub hidden_buffer: &'a MTLBuffer,
    pub partial_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
    pub w13_all: &'a MTLBuffer,
    pub w2_all: &'a MTLBuffer,
    pub up_biases: &'a MTLBuffer,
    pub down_biases: &'a MTLBuffer,
    pub tile_counts: &'a MTLBuffer,
    pub tile_offsets: &'a MTLBuffer,
    pub tile_map: &'a MTLBuffer,
    pub total_tiles: &'a MTLBuffer,
    pub dispatch_args: &'a MTLBuffer,
    pub total_rows: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub e: usize,
    pub num_tiles_k: u32,
    pub gating_code: u32,
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
    pub data_type: KernelDataType,
}

impl MoeExpertsFusedKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeExpertsError> {
        let dtypes = [
            KernelDataType::Float16,
            KernelDataType::BFloat16,
            KernelDataType::Float32,
        ];
        let mut pipelines = vec![Vec::with_capacity(dtypes.len()); 4];
        for gate in 0u32..4u32 {
            for dtype in &dtypes {
                let dtype_suffix = dtype_suffix(*dtype);
                let fcv = FunctionConstantValues::new();
                fcv.set_constant_value_at_index(
                    &gate as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    30,
                );
                let n_group: u32 = 8;
                fcv.set_constant_value_at_index(
                    &n_group as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    31,
                );
                let kernel_name =
                    format!("moe_fused_expert_mlp_{}", dtype_suffix);
                pipelines[gate as usize].push(
                    ctx.compute_pipeline_state(&kernel_name, Some(&fcv))?,
                );
            }
        }
        Ok(Self {
            pipelines,
            tile_map: MoeTileMapKernel::new(ctx)?,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeExpertsArguments,
    ) -> Result<(), MoeExpertsError> {
        self.tile_map.encode_counts(
            command_buffer,
            &MoeTileCountsArguments {
                offsets_buffer: args.expert_offsets,
                tile_counts_buffer: args.tile_counts,
                e: args.e,
            },
        )?;
        self.tile_map.encode_scan(
            command_buffer,
            &MoeTileScanArguments {
                tile_counts_buffer: args.tile_counts,
                tile_offsets_buffer: args.tile_row_offsets,
                total_tiles_buffer: args.total_tiles,
                e: args.e,
            },
        )?;
        self.tile_map.encode_build_map(
            command_buffer,
            &MoeTileMapBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: args.tile_row_offsets,
                tile_counts: args.tile_counts,
                tile_map: args.tile_map,
                e: args.e,
            },
        )?;
        const N_GROUP: u32 = 8;
        let ntx_u32 = (args.num_tiles_n as u32 + N_GROUP - 1) / N_GROUP;
        self.tile_map.encode_dispatch_args(
            command_buffer,
            &MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_x: ntx_u32,
            },
        )?;
        if args.num_tiles_n == 0 {
            return Ok(());
        }
        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = dtype_index(args.data_type);
        let pipeline = &self.pipelines[gate_idx][dtype_idx];
        let t_u = args.t as u32;
        let dm_u = args.d_model as u32;
        let dff_u = args.d_ff as u32;
        let e_u32 = args.e as u32;
        let gate = args.gating_code as u32;
        let threads_per_threadgroup = MTLSize::new(128, 1, 1);
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(args.x_perm_buffer), 0);
        encoder.set_buffer(1, Some(args.expert_offsets), 0);
        encoder.set_buffer(2, Some(args.w13_all), 0);
        encoder.set_buffer(3, Some(args.w2_all), 0);
        encoder.set_buffer(4, Some(args.y_partial), 0);
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &t_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &dm_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &dff_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &gate as *const u32 as *const _,
        );
        encoder.set_buffer(10, Some(args.up_biases), 0);
        encoder.set_buffer(11, Some(args.down_biases), 0);
        encoder.set_bytes(
            12,
            size_of::<f32>() as u64,
            &args.gate_clip_min as *const f32 as *const _,
        );
        encoder.set_bytes(
            13,
            size_of::<f32>() as u64,
            &args.gate_clip_max as *const f32 as *const _,
        );
        encoder.set_bytes(
            14,
            size_of::<f32>() as u64,
            &args.up_clip_min as *const f32 as *const _,
        );
        encoder.set_bytes(
            15,
            size_of::<f32>() as u64,
            &args.up_clip_max as *const f32 as *const _,
        );
        encoder.set_bytes(
            16,
            size_of::<f32>() as u64,
            &args.silu_alpha as *const f32 as *const _,
        );
        encoder.set_buffer(17, Some(args.tile_row_offsets), 0);
        encoder.set_buffer(18, Some(args.tile_map), 0);
        encoder.set_buffer(19, Some(args.total_tiles), 0);
        let y_base_u32: u32 = 0;
        encoder.set_bytes(
            20,
            size_of::<u32>() as u64,
            &y_base_u32 as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups_indirect(
            args.dispatch_args,
            0,
            threads_per_threadgroup,
        );
        encoder.end_encoding();
        Ok(())
    }
}

impl MoeExpertsTwoPassDecodeKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeExpertsError> {
        let dtypes = [
            KernelDataType::Float16,
            KernelDataType::BFloat16,
            KernelDataType::Float32,
        ];
        let mut pass_a_indirect = vec![Vec::with_capacity(dtypes.len()); 4];
        for gate in 0u32..4u32 {
            for dtype in &dtypes {
                let dtype_suffix = dtype_suffix(*dtype);
                let fcv = FunctionConstantValues::new();
                fcv.set_constant_value_at_index(
                    &gate as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    30,
                );
                let tile_h: u32 = 512;
                fcv.set_constant_value_at_index(
                    &tile_h as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    32,
                );
                let kernel_name = format!(
                    "moe_experts_decode_pass_a_indirect_{}",
                    dtype_suffix
                );
                pass_a_indirect[gate as usize].push(
                    ctx.compute_pipeline_state(&kernel_name, Some(&fcv))?,
                );
            }
        }
        let mut fused_down = Vec::with_capacity(dtypes.len());
        for dtype in &dtypes {
            let dtype_suffix = dtype_suffix(*dtype);
            let kernel_name =
                format!("moe_experts_decode_down_fused_2d_{}", dtype_suffix);
            fused_down.push(ctx.compute_pipeline_state(&kernel_name, None)?);
        }
        Ok(Self {
            pass_a_tile: MoePassATileKernel::new(ctx)?,
            pass_a_indirect,
            fused_down,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeExpertsTwoPassArguments,
    ) -> Result<(), MoeExpertsError> {
        if args.total_rows == 0 {
            return Ok(());
        }
        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = dtype_index(args.data_type);
        const BLOCK_M: u32 = 4;
        let h_blocks = (args.d_ff as u32 + BLOCK_M - 1) / BLOCK_M;
        self.pass_a_tile.encode_counts(
            command_buffer,
            &MoePassATileCountsArguments {
                expert_offsets: args.expert_offsets,
                tile_counts: args.tile_counts,
                e: args.e,
                h_blocks,
            },
        )?;
        self.pass_a_tile.encode_scan(
            command_buffer,
            &MoePassATileScanArguments {
                tile_counts: args.tile_counts,
                tile_offsets: args.tile_offsets,
                total_tiles: args.total_tiles,
                e: args.e,
            },
        )?;
        self.pass_a_tile.encode_row_map(
            command_buffer,
            &MoePassARowMapArguments {
                expert_offsets: args.expert_offsets,
                row_expert_map: args.row_expert_map,
                total_rows: args.total_rows,
                e: args.e,
            },
        )?;
        self.pass_a_tile.encode_build_map(
            command_buffer,
            &MoePassATileBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: args.tile_offsets,
                row_expert_map: args.row_expert_map,
                tile_map: args.tile_map,
                total_rows: args.total_rows,
                h_blocks,
            },
        )?;
        self.pass_a_tile.encode_dispatch_args(
            command_buffer,
            &MoePassATileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_y: 1,
            },
        )?;
        let d_model_u32 = args.d_model as u32;
        let d_ff_u32 = args.d_ff as u32;
        let e_u32 = args.e as u32;
        let pass_a_pipeline = &self.pass_a_indirect[gate_idx][dtype_idx];
        let encoder_a = command_buffer.new_compute_command_encoder();
        encoder_a.set_compute_pipeline_state(pass_a_pipeline);
        encoder_a.set_buffer(0, Some(args.x_perm_buffer), 0);
        encoder_a.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_a.set_buffer(2, Some(args.w13_all), 0);
        encoder_a.set_buffer(3, Some(args.hidden_buffer), 0);
        encoder_a.set_buffer(4, Some(args.up_biases), 0);
        encoder_a.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            7,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            8,
            size_of::<f32>() as u64,
            &args.gate_clip_min as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            9,
            size_of::<f32>() as u64,
            &args.gate_clip_max as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            10,
            size_of::<f32>() as u64,
            &args.up_clip_min as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            11,
            size_of::<f32>() as u64,
            &args.up_clip_max as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            12,
            size_of::<f32>() as u64,
            &args.silu_alpha as *const f32 as *const _,
        );
        encoder_a.set_buffer(13, Some(args.tile_map), 0);
        encoder_a.dispatch_thread_groups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(128, 1, 1),
        );
        encoder_a.end_encoding();
        let total_rows_u32 = args.total_rows as u32;
        let pass_b_pipeline = &self.fused_down[dtype_idx];
        let encoder_b = command_buffer.new_compute_command_encoder();
        encoder_b.set_compute_pipeline_state(pass_b_pipeline);
        encoder_b.set_buffer(0, Some(args.hidden_buffer), 0);
        encoder_b.set_buffer(1, Some(args.row_expert_map), 0);
        encoder_b.set_buffer(2, Some(args.w2_all), 0);
        encoder_b.set_buffer(3, Some(args.down_biases), 0);
        encoder_b.set_buffer(4, Some(args.output_buffer), 0);
        encoder_b.set_bytes(
            5,
            size_of::<u32>() as u64,
            &total_rows_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            7,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            8,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        const SIMDGROUPS_PER_TG: u32 = 8;
        const THREADS_PER_TG: u32 = 256;
        let col_blocks =
            (args.d_model as u32 + SIMDGROUPS_PER_TG - 1) / SIMDGROUPS_PER_TG;
        encoder_b.dispatch_thread_groups(
            MTLSize::new(col_blocks as u64, args.total_rows as u64, 1),
            MTLSize::new(THREADS_PER_TG as u64, 1, 1),
        );
        encoder_b.end_encoding();
        Ok(())
    }
}

impl MoeExpertsTwoPassPrefillKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeExpertsError> {
        let dtypes = [
            KernelDataType::Float16,
            KernelDataType::BFloat16,
            KernelDataType::Float32,
        ];
        let mut pass_a_indirect = vec![Vec::with_capacity(dtypes.len()); 4];
        for gate in 0u32..4u32 {
            for dtype in &dtypes {
                let dtype_suffix = dtype_suffix(*dtype);
                let fcv = FunctionConstantValues::new();
                fcv.set_constant_value_at_index(
                    &gate as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    30,
                );
                let kernel_name = format!(
                    "moe_two_pass_prefill_pass_a_indirect_{}",
                    dtype_suffix
                );
                pass_a_indirect[gate as usize].push(
                    ctx.compute_pipeline_state(&kernel_name, Some(&fcv))?,
                );
            }
        }
        let mut pass_b_indirect = Vec::with_capacity(dtypes.len());
        for dtype in &dtypes {
            let dtype_suffix = dtype_suffix(*dtype);
            let kernel_name = format!(
                "moe_two_pass_prefill_pass_b_indirect_{}",
                dtype_suffix
            );
            pass_b_indirect
                .push(ctx.compute_pipeline_state(&kernel_name, None)?);
        }
        Ok(Self {
            tile_map: MoeTileMapKernel::new(ctx)?,
            pass_a_indirect,
            pass_b_indirect,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeExpertsTwoPassArguments,
    ) -> Result<(), MoeExpertsError> {
        if args.total_rows == 0 {
            return Ok(());
        }
        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = dtype_index(args.data_type);
        let dtype_size = match args.data_type {
            KernelDataType::BFloat16 | KernelDataType::Float16 => 2,
            KernelDataType::Float32 => 4,
        };
        let hidden_bytes = (args.total_rows * args.d_ff * dtype_size) as u64;
        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.fill_buffer(
            args.hidden_buffer,
            metal::NSRange::new(0, hidden_bytes),
            0,
        );
        blit_encoder.end_encoding();
        self.tile_map.encode_counts(
            command_buffer,
            &MoeTileCountsArguments {
                offsets_buffer: args.expert_offsets,
                tile_counts_buffer: args.tile_counts,
                e: args.e,
            },
        )?;
        self.tile_map.encode_scan(
            command_buffer,
            &MoeTileScanArguments {
                tile_counts_buffer: args.tile_counts,
                tile_offsets_buffer: args.tile_offsets,
                total_tiles_buffer: args.total_tiles,
                e: args.e,
            },
        )?;
        self.tile_map.encode_build_map(
            command_buffer,
            &MoeTileMapBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: args.tile_offsets,
                tile_counts: args.tile_counts,
                tile_map: args.tile_map,
                e: args.e,
            },
        )?;
        let d_model_u32 = args.d_model as u32;
        let d_ff_u32 = args.d_ff as u32;
        let e_u32 = args.e as u32;
        const COL_TILE_FF: u32 = 64;
        const COL_TILE_MODEL: u32 = 64;
        let n_tiles_ff = if d_ff_u32 == 0 {
            0
        } else {
            (d_ff_u32 + COL_TILE_FF - 1) / COL_TILE_FF
        };
        let n_tiles_model = if d_model_u32 == 0 {
            0
        } else {
            (d_model_u32 + COL_TILE_MODEL - 1) / COL_TILE_MODEL
        };
        self.tile_map.encode_dispatch_args(
            command_buffer,
            &MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_x: n_tiles_ff,
            },
        )?;
        let pass_a_pipeline = &self.pass_a_indirect[gate_idx][dtype_idx];
        let encoder_a = command_buffer.new_compute_command_encoder();
        encoder_a.set_compute_pipeline_state(pass_a_pipeline);
        encoder_a.set_buffer(0, Some(args.x_perm_buffer), 0);
        encoder_a.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_a.set_buffer(2, Some(args.w13_all), 0);
        encoder_a.set_buffer(3, Some(args.up_biases), 0);
        encoder_a.set_buffer(4, Some(args.hidden_buffer), 0);
        encoder_a.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            7,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            8,
            size_of::<f32>() as u64,
            &args.gate_clip_min as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            9,
            size_of::<f32>() as u64,
            &args.gate_clip_max as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            10,
            size_of::<f32>() as u64,
            &args.up_clip_min as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            11,
            size_of::<f32>() as u64,
            &args.up_clip_max as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            12,
            size_of::<f32>() as u64,
            &args.silu_alpha as *const f32 as *const _,
        );
        encoder_a.set_buffer(13, Some(args.tile_map), 0);
        const SIMDGROUPS_PER_TG: u32 = 8;
        const THREADS_PER_TG: u32 = SIMDGROUPS_PER_TG * 32;
        encoder_a.dispatch_thread_groups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(THREADS_PER_TG as u64, 1, 1),
        );
        encoder_a.end_encoding();
        self.tile_map.encode_dispatch_args(
            command_buffer,
            &MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_x: n_tiles_model,
            },
        )?;
        let pass_b_pipeline = &self.pass_b_indirect[dtype_idx];
        let encoder_b = command_buffer.new_compute_command_encoder();
        encoder_b.set_compute_pipeline_state(pass_b_pipeline);
        encoder_b.set_buffer(0, Some(args.hidden_buffer), 0);
        encoder_b.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_b.set_buffer(2, Some(args.w2_all), 0);
        encoder_b.set_buffer(3, Some(args.down_biases), 0);
        encoder_b.set_buffer(4, Some(args.output_buffer), 0);
        encoder_b.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            7,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_b.set_buffer(8, Some(args.tile_map), 0);
        encoder_b.dispatch_thread_groups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(THREADS_PER_TG as u64, 1, 1),
        );
        encoder_b.end_encoding();
        Ok(())
    }
}

pub struct MoeScatterKernels {
    pipeline_bases: MTLComputePipelineState,
    pipeline_scatter_f16: MTLComputePipelineState,
    pipeline_scatter_f32: MTLComputePipelineState,
    pipeline_scatter_bf16: MTLComputePipelineState,
    // map variants
    pipeline_scatter_map_f16: MTLComputePipelineState,
    pipeline_scatter_map_f32: MTLComputePipelineState,
    pipeline_scatter_map_bf16: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeBlockBasesArguments<'a> {
    pub partials_buffer: &'a MTLBuffer, // [num_blocks * num_tiles * 512]
    pub block_bases_buffer: &'a MTLBuffer, // same shape as partials
    pub block_alloc_buffer: &'a MTLBuffer, // [num_blocks * num_tiles * 512]
    pub e: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

#[derive(Debug)]
pub struct MoeScatterArguments<'a> {
    pub topk_ids_buffer: &'a MTLBuffer,
    pub topk_probs_buffer: &'a MTLBuffer,
    pub offsets_buffer: &'a MTLBuffer,
    pub block_bases_buffer: &'a MTLBuffer,
    pub block_alloc_buffer: &'a MTLBuffer,
    pub out_ids_buffer: &'a MTLBuffer,
    pub out_probs_buffer: &'a MTLBuffer,
    pub t: usize,
    pub e: usize,
    pub k: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

#[derive(Debug)]
pub struct MoeScatterWithMapArguments<'a> {
    pub base: MoeScatterArguments<'a>,
    pub tok2row_buffer: &'a MTLBuffer, // [T*K] int32, initialized to -1
}

impl MoeScatterKernels {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeScatterError> {
        let pipeline_bases = mtl_context
            .compute_pipeline_state("moe_block_bases_from_partials", None)?;
        let pipeline_scatter_f16 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_f16", None)?;
        let pipeline_scatter_f32 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_f32", None)?;
        let pipeline_scatter_bf16 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_bf16", None)?;
        let pipeline_scatter_map_f16 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_map_f16", None)?;
        let pipeline_scatter_map_f32 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_map_f32", None)?;
        let pipeline_scatter_map_bf16 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_map_bf16", None)?;

        Ok(Self {
            pipeline_bases,
            pipeline_scatter_f16,
            pipeline_scatter_f32,
            pipeline_scatter_bf16,
            pipeline_scatter_map_f16,
            pipeline_scatter_map_f32,
            pipeline_scatter_map_bf16,
        })
    }

    pub fn encode_block_bases(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeBlockBasesArguments,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&self.pipeline_bases);
        compute_encoder.set_buffer(0, Some(args.partials_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.block_bases_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.block_alloc_buffer), 0);

        let e_u32 = args.e as u32;
        let nb_u32 = args.num_blocks as u32;
        let nt_u32 = args.num_tiles as u32;
        let cap_u32: u32 = 0;
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &nb_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &nt_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &cap_u32 as *const u32 as *const std::ffi::c_void,
        );

        let total_entries = args.num_tiles * 512usize;
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new(((total_entries + 255) / 256) as u64, 1, 1);
        if total_entries > 0 {
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scatter(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeScatterArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder = command_buffer.new_compute_command_encoder();
        // Select pipeline based on dtype
        match dtype {
            KernelDataType::Float16 => {
                compute_encoder
                    .set_compute_pipeline_state(&self.pipeline_scatter_f16);
            },
            KernelDataType::Float32 => {
                compute_encoder
                    .set_compute_pipeline_state(&self.pipeline_scatter_f32);
            },
            KernelDataType::BFloat16 => {
                compute_encoder
                    .set_compute_pipeline_state(&self.pipeline_scatter_bf16);
            },
        }
        compute_encoder.set_buffer(0, Some(args.topk_ids_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.topk_probs_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.offsets_buffer), 0);
        compute_encoder.set_buffer(3, Some(args.block_bases_buffer), 0);
        compute_encoder.set_buffer(4, Some(args.block_alloc_buffer), 0);
        compute_encoder.set_buffer(5, Some(args.out_ids_buffer), 0);
        compute_encoder.set_buffer(6, Some(args.out_probs_buffer), 0);
        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;
        let nb_u32 = args.num_blocks as u32;
        let nt_u32 = args.num_tiles as u32;
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &nb_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            &nt_u32 as *const u32 as *const std::ffi::c_void,
        );

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new(args.num_blocks as u64, 1, 1);
        if args.num_blocks > 0 {
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scatter_with_map(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeScatterWithMapArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder = command_buffer.new_compute_command_encoder();
        let pipeline = match dtype {
            KernelDataType::Float16 => &self.pipeline_scatter_map_f16,
            KernelDataType::Float32 => &self.pipeline_scatter_map_f32,
            KernelDataType::BFloat16 => &self.pipeline_scatter_map_bf16,
        };
        compute_encoder.set_compute_pipeline_state(pipeline);
        let base = &args.base;
        compute_encoder.set_buffer(0, Some(base.topk_ids_buffer), 0);
        compute_encoder.set_buffer(1, Some(base.topk_probs_buffer), 0);
        compute_encoder.set_buffer(2, Some(base.offsets_buffer), 0);
        compute_encoder.set_buffer(3, Some(base.block_bases_buffer), 0);
        compute_encoder.set_buffer(4, Some(base.block_alloc_buffer), 0);
        compute_encoder.set_buffer(5, Some(base.out_ids_buffer), 0);
        compute_encoder.set_buffer(6, Some(base.out_probs_buffer), 0);
        let t_u32 = base.t as u32;
        let e_u32 = base.e as u32;
        let k_u32 = base.k as u32;
        let nb_u32 = base.num_blocks as u32;
        let nt_u32 = base.num_tiles as u32;
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &nb_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            &nt_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_buffer(12, Some(args.tok2row_buffer), 0);

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new(base.num_blocks as u64, 1, 1);
        if base.num_blocks > 0 {
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }
}

impl MoeBucketCountsKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeBucketCountsError> {
        let pipeline_partials =
            mtl_context.compute_pipeline_state("moe_bucket_partials", None)?;
        let pipeline_reduce = mtl_context
            .compute_pipeline_state("moe_bucket_reduce_partials", None)?;
        Ok(Self {
            pipeline_partials,
            pipeline_reduce,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeBucketCountsArguments,
    ) -> Result<(), MoeBucketCountsError> {
        if args.k == 0 || args.e == 0 {
            return Ok(());
        }
        if args.t == 0 {
            return Ok(());
        }
        if args.e < 1 {
            return Err(MoeBucketCountsError::InvalidDimensions {
                t: args.t,
                e: args.e,
                k: args.k,
            });
        }

        let compute_encoder = command_buffer.new_compute_command_encoder();
        // Pass A: partials
        compute_encoder.set_compute_pipeline_state(&self.pipeline_partials);
        compute_encoder.set_buffer(0, Some(args.topk_ids_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.partials_buffer), 0);

        // Compute sizes
        let num_blocks = ((args.t + 255) / 256) as u32;
        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &num_blocks as *const u32 as *const std::ffi::c_void,
        );

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        if num_blocks > 0 {
            let tg = MTLSize::new(num_blocks as u64, 1, 1);
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }

        // Pass B: reduce
        compute_encoder.set_compute_pipeline_state(&self.pipeline_reduce);
        compute_encoder.set_buffer(0, Some(args.partials_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.counts_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &num_blocks as *const u32 as *const std::ffi::c_void,
        );
        let tg2 = MTLSize::new(((args.e + 255) / 256) as u64, 1, 1);
        if args.e > 0 {
            compute_encoder
                .dispatch_thread_groups(tg2, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }
}

impl MoeTopKKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeTopKError> {
        let pipeline_f16 =
            mtl_context.compute_pipeline_state("moe_topk_select_f16", None)?;
        let pipeline_f32 =
            mtl_context.compute_pipeline_state("moe_topk_select_f32", None)?;
        let pipeline_bf16 =
            mtl_context.compute_pipeline_state("moe_topk_select_bf16", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_f32,
            pipeline_bf16,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        dtype: KernelDataType,
        args: MoeTopKArguments,
    ) -> Result<(), MoeTopKError> {
        if args.k == 0 || args.e == 0 || args.t == 0 {
            // No-op for empty work; allow t==0 silently
            if args.t == 0 {
                return Ok(());
            }
        }
        if args.e < args.k {
            return Err(MoeTopKError::InvalidDimensions {
                t: args.t,
                e: args.e,
                k: args.k,
            });
        }

        let compute_encoder = command_buffer.new_compute_command_encoder();
        match dtype {
            KernelDataType::Float16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f16)
            },
            KernelDataType::Float32 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f32)
            },
            KernelDataType::BFloat16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_bf16)
            },
        }

        compute_encoder.set_buffer(0, Some(args.logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.topk_ids_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.topk_probs_buffer), 0);

        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;
        let renorm_u32: u32 = if args.renorm {
            1
        } else {
            0
        };

        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &renorm_u32 as *const u32 as *const std::ffi::c_void,
        );

        // Launch
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let num_threadgroups = if args.t == 0 {
            0
        } else {
            (args.t + 255) / 256
        } as u64;
        if num_threadgroups > 0 {
            let threadgroups = MTLSize::new(num_threadgroups, 1, 1);
            compute_encoder
                .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();

        Ok(())
    }
}

// ---- Finalize Kernel ----
#[derive(Debug, thiserror::Error)]
pub enum MoeFinalizeError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeFinalizeKernel {
    pipeline_f16: MTLComputePipelineState,
    pipeline_bf16: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeFinalizeArguments<'a> {
    pub tok2row_buffer: &'a MTLBuffer, // [T*K] i32
    pub probs_buffer: &'a MTLBuffer,   // [T*K] f16/bf16
    pub y_partial_buffer: &'a MTLBuffer, // [sum_k, d_model] f16
    pub y_out_buffer: &'a MTLBuffer,   // [T, d_model] f16
    pub t: usize,
    pub d_model: usize,
    pub k: usize,
}

impl MoeFinalizeKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeFinalizeError> {
        let pipeline_f16 =
            ctx.compute_pipeline_state("moe_finalize_f16", None)?;
        let pipeline_bf16 =
            ctx.compute_pipeline_state("moe_finalize_bf16", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_bf16,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeFinalizeArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeFinalizeError> {
        let encoder = command_buffer.new_compute_command_encoder();
        match dtype {
            KernelDataType::Float16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_f16);
            },
            KernelDataType::BFloat16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_bf16);
            },
            KernelDataType::Float32 => {
                // Not used for finalize in v1
                encoder.set_compute_pipeline_state(&self.pipeline_f16);
            },
        }
        encoder.set_buffer(0, Some(args.tok2row_buffer), 0);
        encoder.set_buffer(1, Some(args.probs_buffer), 0);
        encoder.set_buffer(2, Some(args.y_partial_buffer), 0);
        encoder.set_buffer(3, Some(args.y_out_buffer), 0);
        let t_u = args.t as u32;
        let dm_u = args.d_model as u32;
        let k_u = args.k as u32;
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &t_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &dm_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &k_u as *const u32 as *const _,
        );

        // Launch tiles over (N tiles, M tiles)
        const BM: usize = 32;
        const BN: usize = 64;
        let num_tiles_n = (args.d_model + BN - 1) / BN;
        let num_tiles_m = (args.t + BM - 1) / BM;
        let threads_per_threadgroup = MTLSize::new(128, 1, 1); // BM * BN * 32 = 1 * 4 * 32
        if num_tiles_m > 0 && num_tiles_n > 0 {
            let threadgroups =
                MTLSize::new(num_tiles_n as u64, num_tiles_m as u64, 1);
            encoder
                .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        }
        encoder.end_encoding();
        Ok(())
    }
}
