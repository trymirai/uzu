use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

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
