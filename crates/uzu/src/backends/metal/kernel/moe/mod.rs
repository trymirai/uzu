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

/// Arguments for standalone router encoder
pub struct RouterEncoderArgs<'a> {
    pub input_buffer: &'a MTLBuffer,  // [T, d_model]
    pub weight_buffer: &'a MTLBuffer, // [E, d_model]
    pub bias_buffer: &'a MTLBuffer,   // [E]
    pub output_buffer: &'a MTLBuffer, // [T, E]
    pub t: usize,
    pub d_model: usize,
    pub e: usize,
}

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

/// Encode MoE gather kernel (standalone, reusable)
pub fn encode_moe_gather(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    dtype: KernelDataType,
    args: &MoeGatherArguments,
) -> Result<(), MTLError> {
    let kernel_name = match dtype {
        KernelDataType::BFloat16 => "moe_gather_x_perm_bf16",
        KernelDataType::Float16 => "moe_gather_x_perm_f16",
        KernelDataType::Float32 => "moe_gather_x_perm_f32",
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
        encoder.dispatch_thread_groups(
            MTLSize::new(((max_rows + 255) / 256) as u64, 1, 1),
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
        encoder.dispatch_thread_groups(
            MTLSize::new(((max_rows + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
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
            ctx.compute_pipeline_state("moe_gather_x_perm_bf16", None)?;
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
        let threadgroups = MTLSize::new(((max_rows + 255) / 256) as u64, 1, 1);
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
}

pub struct MoeExpertsKernel {
    pipelines: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype] - tiled version
    gemv_pipelines: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype] - decode-specialized GEMV
    two_pass_pass_a: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype][transposed]
    two_pass_partial: Vec<MTLComputePipelineState>,     // [dtype]
    two_pass_reduce: Vec<MTLComputePipelineState>,      // [dtype]
    tile_counts_pipeline: MTLComputePipelineState,
    tile_scan_pipeline: MTLComputePipelineState,
    tile_build_map_pipeline: MTLComputePipelineState,
    write_dispatch_args_pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeExpertsArguments<'a> {
    pub x_perm_buffer: &'a MTLBuffer, // [sum_k, d_model]
    pub expert_offsets: &'a MTLBuffer, // [E+1]
    pub w13_all: &'a MTLBuffer,       // [E*d_model*2*d_ff]
    pub w2_all: &'a MTLBuffer,        // [E*d_model*d_ff]
    pub y_partial: &'a MTLBuffer,     // [sum_k,d_model]
    pub up_biases: &'a MTLBuffer,     // [E*2*d_ff] if fused, [E*d_ff] otherwise
    pub down_biases: &'a MTLBuffer,   // [E*d_model]
    pub tile_counts: &'a MTLBuffer,   // [E]
    pub tile_row_offsets: &'a MTLBuffer, // [E+1]
    pub tile_map: &'a MTLBuffer,      // [max_tiles * 3]
    pub total_tiles: &'a MTLBuffer,   // [2]
    pub dispatch_args: &'a MTLBuffer, // [3] u32 for indirect dispatch
    pub num_tiles_n: usize,
    pub t: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub e: usize,
    pub k: usize,         // num_experts_per_token
    pub gating_code: u32, // 0=GELU,1=SiLU,2=SwiGLU,3=GEGLU
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
    pub data_type: KernelDataType,
}

#[derive(Debug)]
pub struct MoeExpertsTwoPassArguments<'a> {
    pub x_perm_buffer: &'a MTLBuffer,
    pub expert_offsets: &'a MTLBuffer,
    pub hidden_buffer: &'a MTLBuffer,
    pub partial_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
    pub w13_all: &'a MTLBuffer,
    pub w2_all: &'a MTLBuffer,
    pub up_biases: &'a MTLBuffer,
    pub down_biases: &'a MTLBuffer,
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

impl MoeExpertsKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeExpertsError> {
        let dtypes = [
            KernelDataType::Float16,
            KernelDataType::BFloat16,
            KernelDataType::Float32,
        ];

        let mut pipelines: Vec<Vec<MTLComputePipelineState>> =
            vec![Vec::with_capacity(dtypes.len()); 4];
        let mut gemv_pipelines: Vec<Vec<MTLComputePipelineState>> =
            vec![Vec::with_capacity(dtypes.len()); 4];
        let mut two_pass_pass_a: Vec<Vec<MTLComputePipelineState>> =
            vec![Vec::with_capacity(dtypes.len()); 4];
        let mut two_pass_partial: Vec<MTLComputePipelineState> =
            Vec::with_capacity(dtypes.len());
        let mut two_pass_reduce: Vec<MTLComputePipelineState> =
            Vec::with_capacity(dtypes.len());
        let two_pass_k_tile: u32 = 64;

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
                let pipeline =
                    ctx.compute_pipeline_state(&kernel_name, Some(&fcv))?;
                pipelines[gate as usize].push(pipeline);

                let gemv_fcv = FunctionConstantValues::new();
                gemv_fcv.set_constant_value_at_index(
                    &gate as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    30,
                );
                let tile_h: u32 = 512;
                gemv_fcv.set_constant_value_at_index(
                    &tile_h as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    32,
                );
                let gemv_kernel =
                    format!("moe_experts_decode_fused_gemv_{}", dtype_suffix);
                let gemv_pipeline =
                    ctx.compute_pipeline_state(&gemv_kernel, Some(&gemv_fcv))?;
                gemv_pipelines[gate as usize].push(gemv_pipeline);

                let kernel =
                    format!("moe_experts_decode_pass_a_{}", dtype_suffix);
                let pass_a_fcv = FunctionConstantValues::new();
                pass_a_fcv.set_constant_value_at_index(
                    &gate as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    30,
                );
                let tile_h: u32 = 512;
                pass_a_fcv.set_constant_value_at_index(
                    &tile_h as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    32,
                );
                pass_a_fcv.set_constant_value_at_index(
                    &two_pass_k_tile as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    33,
                );
                let pass_a_pipeline =
                    ctx.compute_pipeline_state(&kernel, Some(&pass_a_fcv))?;
                two_pass_pass_a[gate as usize].push(pass_a_pipeline);
            }
        }

        for dtype in &dtypes {
            let dtype_suffix = dtype_suffix(*dtype);

            let partial_fcv = FunctionConstantValues::new();
            let gate_zero: u32 = 0;
            partial_fcv.set_constant_value_at_index(
                &gate_zero as *const u32 as *const std::ffi::c_void,
                MTLDataType::UInt,
                30,
            );
            let tile_h_one: u32 = 1;
            partial_fcv.set_constant_value_at_index(
                &tile_h_one as *const u32 as *const std::ffi::c_void,
                MTLDataType::UInt,
                32,
            );
            partial_fcv.set_constant_value_at_index(
                &two_pass_k_tile as *const u32 as *const std::ffi::c_void,
                MTLDataType::UInt,
                33,
            );
            let partial_kernel =
                format!("moe_experts_decode_down_partial_{}", dtype_suffix);
            two_pass_partial.push(
                ctx.compute_pipeline_state(
                    &partial_kernel,
                    Some(&partial_fcv),
                )?,
            );

            let reduce_fcv = FunctionConstantValues::new();
            reduce_fcv.set_constant_value_at_index(
                &gate_zero as *const u32 as *const std::ffi::c_void,
                MTLDataType::UInt,
                30,
            );
            reduce_fcv.set_constant_value_at_index(
                &tile_h_one as *const u32 as *const std::ffi::c_void,
                MTLDataType::UInt,
                32,
            );
            reduce_fcv.set_constant_value_at_index(
                &two_pass_k_tile as *const u32 as *const std::ffi::c_void,
                MTLDataType::UInt,
                33,
            );
            let reduce_kernel =
                format!("moe_experts_decode_down_reduce_{}", dtype_suffix);
            two_pass_reduce.push(
                ctx.compute_pipeline_state(&reduce_kernel, Some(&reduce_fcv))?,
            );
        }

        let tile_counts =
            ctx.compute_pipeline_state("moe_tile_counts", None)?;
        let tile_scan = ctx.compute_pipeline_state("moe_tile_scan", None)?;
        let tile_build =
            ctx.compute_pipeline_state("moe_build_tile_map", None)?;
        let write_dispatch_args =
            ctx.compute_pipeline_state("moe_write_dispatch_args", None)?;

        Ok(Self {
            pipelines,
            gemv_pipelines,
            two_pass_pass_a,
            two_pass_partial,
            two_pass_reduce,
            tile_counts_pipeline: tile_counts,
            tile_scan_pipeline: tile_scan,
            tile_build_map_pipeline: tile_build,
            write_dispatch_args_pipeline: write_dispatch_args,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeExpertsArguments,
    ) -> Result<(), MoeExpertsError> {
        // Otherwise use tiled Multi-N fusion path
        let e_u32 = args.e as u32;

        // Pass A: per-expert tile counts
        let encoder_a = command_buffer.new_compute_command_encoder();
        encoder_a.set_compute_pipeline_state(&self.tile_counts_pipeline);
        encoder_a.set_buffer(0, Some(args.expert_offsets), 0);
        encoder_a.set_buffer(1, Some(args.tile_counts), 0);
        encoder_a.set_bytes(
            2,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_a.dispatch_thread_groups(
            MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder_a.end_encoding();

        // Pass B: exclusive scan for tile offsets + total tile count
        let encoder_b = command_buffer.new_compute_command_encoder();
        encoder_b.set_compute_pipeline_state(&self.tile_scan_pipeline);
        encoder_b.set_buffer(0, Some(args.tile_counts), 0);
        encoder_b.set_buffer(1, Some(args.tile_row_offsets), 0);
        encoder_b.set_buffer(2, Some(args.total_tiles), 0);
        encoder_b.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_b.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
        encoder_b.end_encoding();

        // Pass C: flatten into tile_map descriptors
        let encoder_c = command_buffer.new_compute_command_encoder();
        encoder_c.set_compute_pipeline_state(&self.tile_build_map_pipeline);
        encoder_c.set_buffer(0, Some(args.expert_offsets), 0);
        encoder_c.set_buffer(1, Some(args.tile_row_offsets), 0);
        encoder_c.set_buffer(2, Some(args.tile_counts), 0);
        encoder_c.set_buffer(3, Some(args.tile_map), 0);
        encoder_c.set_bytes(
            4,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_c.dispatch_thread_groups(
            MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder_c.end_encoding();

        // Tiny pass: write MTLDispatchThreadgroupsIndirectArguments {x, y, z}
        let encoder_w = command_buffer.new_compute_command_encoder();
        encoder_w
            .set_compute_pipeline_state(&self.write_dispatch_args_pipeline);
        encoder_w.set_buffer(0, Some(args.total_tiles), 0);
        encoder_w.set_buffer(1, Some(args.dispatch_args), 0);
        // Multi-N fusion: divide num_tiles_n by N_GROUP for dispatch
        const N_GROUP: u32 = 8; // Hardcoded for now, will parameterize later
        let ntx_u32 = (args.num_tiles_n as u32 + N_GROUP - 1) / N_GROUP;
        encoder_w.set_bytes(
            2,
            size_of::<u32>() as u64,
            &ntx_u32 as *const u32 as *const _,
        );
        encoder_w.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1, 1, 1),
        );
        encoder_w.end_encoding();

        // Experts kernel: deterministic 2D dispatch without atomics
        let num_tiles_n = args.num_tiles_n;
        if num_tiles_n == 0 {
            return Ok(());
        }
        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = match args.data_type {
            KernelDataType::Float16 => 0usize,
            KernelDataType::BFloat16 => 1usize,
            KernelDataType::Float32 => 2usize,
        };
        let pipeline = &self.pipelines[gate_idx][dtype_idx];
        let t_u = args.t as u32;
        let dm_u = args.d_model as u32;
        let dff_u = args.d_ff as u32;
        let gate = args.gating_code as u32;
        let threads_per_threadgroup = MTLSize::new(128, 1, 1);
        let encoder_e = command_buffer.new_compute_command_encoder();
        encoder_e.set_compute_pipeline_state(pipeline);
        encoder_e.set_buffer(0, Some(args.x_perm_buffer), 0);
        encoder_e.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_e.set_buffer(2, Some(args.w13_all), 0);
        encoder_e.set_buffer(3, Some(args.w2_all), 0);
        encoder_e.set_buffer(4, Some(args.y_partial), 0);
        encoder_e.set_bytes(
            5,
            size_of::<u32>() as u64,
            &t_u as *const u32 as *const _,
        );
        encoder_e.set_bytes(
            6,
            size_of::<u32>() as u64,
            &dm_u as *const u32 as *const _,
        );
        encoder_e.set_bytes(
            7,
            size_of::<u32>() as u64,
            &dff_u as *const u32 as *const _,
        );
        encoder_e.set_bytes(
            8,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_e.set_bytes(
            9,
            size_of::<u32>() as u64,
            &gate as *const u32 as *const _,
        );
        encoder_e.set_buffer(10, Some(args.up_biases), 0);
        encoder_e.set_buffer(11, Some(args.down_biases), 0);
        encoder_e.set_bytes(
            12,
            size_of::<f32>() as u64,
            &args.gate_clip_min as *const f32 as *const _,
        );
        encoder_e.set_bytes(
            13,
            size_of::<f32>() as u64,
            &args.gate_clip_max as *const f32 as *const _,
        );
        encoder_e.set_bytes(
            14,
            size_of::<f32>() as u64,
            &args.up_clip_min as *const f32 as *const _,
        );
        encoder_e.set_bytes(
            15,
            size_of::<f32>() as u64,
            &args.up_clip_max as *const f32 as *const _,
        );
        encoder_e.set_bytes(
            16,
            size_of::<f32>() as u64,
            &args.silu_alpha as *const f32 as *const _,
        );
        encoder_e.set_buffer(17, Some(args.tile_row_offsets), 0);
        encoder_e.set_buffer(18, Some(args.tile_map), 0);
        encoder_e.set_buffer(19, Some(args.total_tiles), 0);

        let y_base_u32: u32 = 0;
        encoder_e.set_bytes(
            20,
            size_of::<u32>() as u64,
            &y_base_u32 as *const u32 as *const _,
        );
        encoder_e.dispatch_thread_groups_indirect(
            args.dispatch_args,
            0,
            threads_per_threadgroup,
        );
        encoder_e.end_encoding();

        Ok(())
    }

    pub fn encode_two_pass_decode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeExpertsTwoPassArguments,
    ) -> Result<(), MoeExpertsError> {
        if args.total_rows == 0 {
            return Ok(());
        }

        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = dtype_index(args.data_type);
        // Pass A: compute hidden activations per routed token/expert
        let pass_a_pipeline = &self.two_pass_pass_a[gate_idx][dtype_idx];
        let encoder_a = command_buffer.new_compute_command_encoder();
        encoder_a.set_compute_pipeline_state(pass_a_pipeline);
        encoder_a.set_buffer(0, Some(args.x_perm_buffer), 0);
        encoder_a.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_a.set_buffer(2, Some(args.w13_all), 0);
        encoder_a.set_buffer(3, Some(args.hidden_buffer), 0);
        encoder_a.set_buffer(4, Some(args.up_biases), 0);
        let d_model_u32 = args.d_model as u32;
        let d_ff_u32 = args.d_ff as u32;
        let e_u32 = args.e as u32;
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
        encoder_a.dispatch_thread_groups(
            MTLSize::new(1, args.e as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder_a.end_encoding();

        // Pass B: compute split-K partial outputs
        let partial_pipeline = &self.two_pass_partial[dtype_idx];
        let encoder_p = command_buffer.new_compute_command_encoder();
        encoder_p.set_compute_pipeline_state(partial_pipeline);
        encoder_p.set_buffer(0, Some(args.hidden_buffer), 0);
        encoder_p.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_p.set_buffer(2, Some(args.w2_all), 0);
        encoder_p.set_buffer(3, Some(args.partial_buffer), 0);
        let total_rows_u32 = args.total_rows as u32;
        encoder_p.set_bytes(
            4,
            size_of::<u32>() as u64,
            &total_rows_u32 as *const u32 as *const _,
        );
        encoder_p.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_p.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_p.set_bytes(
            7,
            size_of::<u32>() as u64,
            &args.num_tiles_k as *const u32 as *const _,
        );
        let e_u32 = args.e as u32;
        encoder_p.set_bytes(
            8,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        let col_groups = (args.d_model as u32 + 255) / 256;
        encoder_p.dispatch_thread_groups(
            MTLSize::new(
                col_groups as u64,
                args.total_rows as u64,
                args.num_tiles_k as u64,
            ),
            MTLSize::new(256, 1, 1),
        );
        encoder_p.end_encoding();

        // Pass C: reduce partials and add down projections
        let reduce_pipeline = &self.two_pass_reduce[dtype_idx];
        let encoder_r = command_buffer.new_compute_command_encoder();
        encoder_r.set_compute_pipeline_state(reduce_pipeline);
        encoder_r.set_buffer(0, Some(args.partial_buffer), 0);
        encoder_r.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_r.set_buffer(2, Some(args.down_biases), 0);
        encoder_r.set_buffer(3, Some(args.output_buffer), 0);
        encoder_r.set_bytes(
            4,
            size_of::<u32>() as u64,
            &total_rows_u32 as *const u32 as *const _,
        );
        encoder_r.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_r.set_bytes(
            6,
            size_of::<u32>() as u64,
            &args.num_tiles_k as *const u32 as *const _,
        );
        encoder_r.set_bytes(
            7,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        let col_groups = (args.d_model as u32 + 255) / 256;
        encoder_r.dispatch_thread_groups(
            MTLSize::new(col_groups as u64, args.total_rows as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder_r.end_encoding();

        Ok(())
    }

    // Decode-specialized GEMV path for small sum_k (typically T=1, K=1-2)
    fn encode_gemv_decode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeExpertsArguments,
    ) -> Result<(), MoeExpertsError> {
        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = match args.data_type {
            KernelDataType::Float16 => 0usize,
            KernelDataType::BFloat16 => 1usize,
            KernelDataType::Float32 => 2usize,
        };
        let pipeline = &self.gemv_pipelines[gate_idx][dtype_idx];

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        // Set buffers (simplified - no tiling infrastructure needed)
        encoder.set_buffer(0, Some(args.x_perm_buffer), 0);
        encoder.set_buffer(1, Some(args.expert_offsets), 0);
        encoder.set_buffer(2, Some(args.w13_all), 0);
        encoder.set_buffer(3, Some(args.w2_all), 0);
        encoder.set_buffer(4, Some(args.y_partial), 0);

        let t_u = args.t as u32;
        let dm_u = args.d_model as u32;
        let dff_u = args.d_ff as u32;
        let e_u = args.e as u32;
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
            &e_u as *const u32 as *const _,
        );

        encoder.set_buffer(9, Some(args.up_biases), 0);
        encoder.set_buffer(10, Some(args.down_biases), 0);

        encoder.set_bytes(
            11,
            size_of::<f32>() as u64,
            &args.gate_clip_min as *const f32 as *const _,
        );
        encoder.set_bytes(
            12,
            size_of::<f32>() as u64,
            &args.gate_clip_max as *const f32 as *const _,
        );
        encoder.set_bytes(
            13,
            size_of::<f32>() as u64,
            &args.up_clip_min as *const f32 as *const _,
        );
        encoder.set_bytes(
            14,
            size_of::<f32>() as u64,
            &args.up_clip_max as *const f32 as *const _,
        );
        encoder.set_bytes(
            15,
            size_of::<f32>() as u64,
            &args.silu_alpha as *const f32 as *const _,
        );

        // Dispatch: 2D grid (d_model column tiles, E experts)
        // V2 kernel: 256 threads per TG, each thread handles one output column
        const THREADS_PER_TG: u64 = 256;
        const COLS_PER_TG: u64 = THREADS_PER_TG;
        let tg_x = (args.d_model as u64 + COLS_PER_TG - 1) / COLS_PER_TG;
        let tg_y = args.e as u64;

        encoder.dispatch_thread_groups(
            MTLSize::new(tg_x, tg_y, 1),
            MTLSize::new(THREADS_PER_TG, 1, 1),
        );
        encoder.end_encoding();

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
        let threads_per_threadgroup = MTLSize::new(128, 1, 1);
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
