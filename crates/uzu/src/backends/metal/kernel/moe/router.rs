use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

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
