use std::mem::size_of;

use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState, MTLSize};

use crate::backends::metal::{KernelDataType, MTLContext};

use super::{fn_suffix, SSMKernelError};

pub struct Conv1dForwardKernel {
    pipeline: MTLComputePipelineState,
}

pub struct Conv1dForwardArguments<'a> {
    pub x: &'a MTLBuffer,        // buffer(0)
    pub w: &'a MTLBuffer,        // buffer(1)
    pub b: &'a MTLBuffer,        // buffer(2)
    pub y: &'a MTLBuffer,        // buffer(3)
    pub x_strides: [usize; 3],   // buffer(4)
    pub kernel_size: i32,        // buffer(5)
    pub batch_size: usize,
    pub channels: usize,
    pub seq_len: usize,
}

impl Conv1dForwardKernel {
    pub fn new(context: &MTLContext, data_type: KernelDataType) -> Result<Self, SSMKernelError> {
        let fn_name = format!("conv1d_forward_kernel_{}", fn_suffix(data_type));
        let pipeline = context.compute_pipeline_state(&fn_name, None).map_err(SSMKernelError::MetalError)?;
        Ok(Self { pipeline })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: Conv1dForwardArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.w), 0);
        compute_encoder.set_buffer(2, Some(args.b), 0);
        compute_encoder.set_buffer(3, Some(args.y), 0);

        compute_encoder.set_bytes(4, (3 * size_of::<usize>()) as u64, args.x_strides.as_ptr() as *const _);
        compute_encoder.set_bytes(5, size_of::<i32>() as u64, &args.kernel_size as *const i32 as *const _);

        let threads_per_threadgroup = MTLSize { width: 32, height: 1, depth: 1 };
        let total_threads = MTLSize { width: args.batch_size as u64, height: args.channels as u64, depth: args.seq_len as u64 };
        compute_encoder.dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
