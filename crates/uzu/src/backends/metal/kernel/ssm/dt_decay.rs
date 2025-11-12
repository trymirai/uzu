use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef, MTLSize};

use crate::backends::metal::{KernelDataType, MTLContext};

use super::{SSMKernelError, fn_suffix};

pub struct DtDecayKernel {
    pipeline: metal::ComputePipelineState,
}

pub struct DtDecayArguments<'a> {
    pub dt: &'a MTLBuffer,    // buffer(0) [suffix, num_heads]
    pub decay: &'a MTLBuffer, // buffer(1) [suffix, num_heads]
    pub num_heads: usize,
    pub suffix_length: usize,
}

impl DtDecayKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("ssm_dt_decay_kernel_{}", fn_suffix(data_type));
        let pipeline = context
            .compute_pipeline_state(&fn_name, None)
            .map_err(SSMKernelError::MetalError)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: DtDecayArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(0, Some(args.dt), 0);
        compute_encoder.set_buffer(1, Some(args.decay), 0);

        let num_heads = args.num_heads as i32;
        let suffix = args.suffix_length as i32;

        compute_encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &num_heads as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &suffix as *const i32 as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: args.suffix_length as u64,
            height: args.num_heads as u64,
            depth: 1,
        };

        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
