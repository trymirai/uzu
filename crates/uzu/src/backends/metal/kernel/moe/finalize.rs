use std::mem::size_of;

use crate::backends::metal::{
    BufferRef, CommandBufferRef, ComputeEncoderLegacy, ComputePipelineState,
    KernelDataType, MTLCommandBuffer, MTLCommandEncoder, MTLContext, MTLError,
    mtl_size,
};

// ---- Finalize Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeFinalizeError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeFinalizeKernel {
    pipeline_f16: ComputePipelineState,
    pipeline_bf16: ComputePipelineState,
    pipeline_f32: ComputePipelineState,
}

#[derive(Debug)]
pub struct MoeFinalizeArguments<'a> {
    pub tok2row_buffer: BufferRef<'a>, // [T*K] i32
    pub probs_buffer: BufferRef<'a>,   // [T*K] f16/bf16
    pub y_partial_buffer: BufferRef<'a>, // [sum_k, d_model] f16
    pub y_out_buffer: BufferRef<'a>,   // [T, d_model] f16
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
        let pipeline_f32 =
            ctx.compute_pipeline_state("moe_finalize_f32", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_bf16,
            pipeline_f32,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeFinalizeArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeFinalizeError> {
        let encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        match dtype {
            KernelDataType::Float16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_f16);
            },
            KernelDataType::BFloat16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_bf16);
            },
            KernelDataType::Float32 => {
                encoder.set_compute_pipeline_state(&self.pipeline_f32);
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
        let threads_per_threadgroup = mtl_size(128, 1, 1); // BM * BN * 32 = 1 * 4 * 32
        if num_tiles_m > 0 && num_tiles_n > 0 {
            let threadgroups =
                mtl_size(num_tiles_n as u64, num_tiles_m as u64, 1);
            encoder
                .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        }
        encoder.end_encoding();
        Ok(())
    }
}
