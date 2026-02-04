use crate::backends::metal::{
    KernelDataType, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLContext, MTLError,
    MTLSize, ProtocolObject, Retained,
    metal_extensions::ComputeEncoderSetValue,
};

// ---- Gather Permuted Activations Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeGatherError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeGatherKernel {
    pipeline_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[derive(Debug)]
pub struct MoeGatherArguments<'a> {
    pub x_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub bucketed_ids_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub x_perm_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub sumk_buffer: &'a ProtocolObject<dyn MTLBuffer>,
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
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        dtype: KernelDataType,
        args: MoeGatherArguments,
    ) -> Result<(), MoeGatherError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
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

        encoder.set_buffer(Some(args.x_buffer), 0, 0);
        encoder.set_buffer(Some(args.bucketed_ids_buffer), 0, 1);
        encoder.set_buffer(Some(args.x_perm_buffer), 0, 2);
        encoder.set_buffer(Some(args.sumk_buffer), 0, 3);
        let d_model_u32 = args.d_model as u32;
        encoder.set_value(&d_model_u32, 4);

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
                    (max_rows + BF16_ROWS_PER_TG - 1) / BF16_ROWS_PER_TG,
                    1,
                    1,
                )
            },
            _ => MTLSize::new((max_rows + 255) / 256, 1, 1),
        };
        encoder.dispatch_threadgroups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        Ok(())
    }
}
