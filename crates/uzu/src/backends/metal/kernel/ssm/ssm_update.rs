use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{KernelDataType, MTLContext};

pub struct SSMUpdateKernel {
    pipeline: MTLComputePipelineState,
}

pub struct SSMUpdateArguments<'a> {
    pub x: &'a MTLBuffer,          // buffer(0)
    pub dt: &'a MTLBuffer,         // buffer(1)
    pub a: &'a MTLBuffer,          // buffer(2)
    pub b: &'a MTLBuffer,          // buffer(3)
    pub c: &'a MTLBuffer,          // buffer(4)
    pub d: &'a MTLBuffer,          // buffer(5)
    pub z: &'a MTLBuffer,          // buffer(6)
    pub state: &'a MTLBuffer,      // buffer(7)
    pub y: &'a MTLBuffer,          // buffer(8)
    pub next_state: &'a MTLBuffer, // buffer(9)
    pub batch_size: usize,
    pub channels: usize,
}

impl SSMUpdateKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("ssm_update_kernel_{}", fn_suffix(data_type));
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
        args: SSMUpdateArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.dt), 0);
        compute_encoder.set_buffer(2, Some(args.a), 0);
        compute_encoder.set_buffer(3, Some(args.b), 0);
        compute_encoder.set_buffer(4, Some(args.c), 0);
        compute_encoder.set_buffer(5, Some(args.d), 0);
        compute_encoder.set_buffer(6, Some(args.z), 0);
        compute_encoder.set_buffer(7, Some(args.state), 0);
        compute_encoder.set_buffer(8, Some(args.y), 0);
        compute_encoder.set_buffer(9, Some(args.next_state), 0);

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 32,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: args.batch_size as u64,
            height: args.channels as u64,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
