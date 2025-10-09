use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{MTLContext, MTLError};

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
