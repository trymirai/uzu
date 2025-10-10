use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{MTLContext, MTLError};

// ---- Fused Counts + Offsets Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeCountsOffsetsFusedError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Invalid dimensions: T={t}, E={e}, K={k}")]
    InvalidDimensions {
        t: usize,
        e: usize,
        k: usize,
    },
}

pub struct MoeCountsOffsetsFusedKernel {
    pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeCountsOffsetsFusedArguments<'a> {
    pub topk_ids_buffer: &'a MTLBuffer,
    pub counts_buffer: &'a MTLBuffer, // output [E]
    pub offsets_buffer: &'a MTLBuffer, // output [E+1]
    pub sum_k_buffer: &'a MTLBuffer,  // output [1]
    pub partials_buffer: &'a MTLBuffer, // output [num_tiles * 512] (for block_bases)
    pub t: usize,
    pub e: usize,
    pub k: usize,
}

impl MoeCountsOffsetsFusedKernel {
    pub fn new(
        mtl_context: &MTLContext
    ) -> Result<Self, MoeCountsOffsetsFusedError> {
        let pipeline = mtl_context
            .compute_pipeline_state("moe_counts_offsets_fused", None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeCountsOffsetsFusedArguments,
    ) -> Result<(), MoeCountsOffsetsFusedError> {
        if args.k == 0 || args.e == 0 {
            return Ok(());
        }
        if args.t == 0 {
            return Ok(());
        }
        if args.e < 1 {
            return Err(MoeCountsOffsetsFusedError::InvalidDimensions {
                t: args.t,
                e: args.e,
                k: args.k,
            });
        }

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);

        encoder.set_buffer(0, Some(args.topk_ids_buffer), 0);
        encoder.set_buffer(1, Some(args.counts_buffer), 0);
        encoder.set_buffer(2, Some(args.offsets_buffer), 0);
        encoder.set_buffer(3, Some(args.sum_k_buffer), 0);
        encoder.set_buffer(4, Some(args.partials_buffer), 0);

        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;

        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const std::ffi::c_void,
        );

        let threads_per_threadgroup = MTLSize::new(128, 1, 1);
        let tg = MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(tg, threads_per_threadgroup);

        encoder.end_encoding();
        Ok(())
    }
}
