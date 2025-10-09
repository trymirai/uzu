use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

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
