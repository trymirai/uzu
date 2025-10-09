use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{MTLContext, MTLError};

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
