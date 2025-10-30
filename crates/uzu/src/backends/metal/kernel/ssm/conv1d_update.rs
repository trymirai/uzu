use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext};

use super::{SSMKernelError, fn_suffix};

pub struct Conv1dUpdateKernel {
    pipeline: MTLComputePipelineState,
}

pub struct Conv1dUpdateArguments<'a> {
    pub x: &'a MTLBuffer,          // buffer(0)  (b, d)
    pub w: &'a MTLBuffer,          // buffer(1)  (d, k)
    pub b: &'a MTLBuffer,          // buffer(2)  (d)
    pub state: &'a MTLBuffer,      // buffer(3)  (b, d, k-1)
    pub y: &'a MTLBuffer,          // buffer(4)
    pub next_state: &'a MTLBuffer, // buffer(5)
    pub kernel_size: i32,          // buffer(6)
    pub x_strides: [usize; 2],     // buffer(7)
    pub state_strides: [usize; 3], // buffer(8)
    pub batch_size: usize,
    pub channels: usize,
}

impl Conv1dUpdateKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("conv1d_update_kernel_{}", fn_suffix(data_type));
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
        args: Conv1dUpdateArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.w), 0);
        compute_encoder.set_buffer(2, Some(args.b), 0);
        compute_encoder.set_buffer(3, Some(args.state), 0);
        compute_encoder.set_buffer(4, Some(args.y), 0);
        compute_encoder.set_buffer(5, Some(args.next_state), 0);

        compute_encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.kernel_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            (2 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            8,
            (3 * size_of::<usize>()) as u64,
            args.state_strides.as_ptr() as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
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
