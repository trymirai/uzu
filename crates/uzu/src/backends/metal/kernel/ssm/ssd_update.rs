use std::mem::size_of;

use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState, MTLSize};

use crate::backends::metal::{KernelDataType, MTLContext};

use super::{fn_suffix, SSMKernelError};

pub struct SSDUpdateKernel {
    pipeline: MTLComputePipelineState,
}

pub struct SSDUpdateArguments<'a> {
    pub x: &'a MTLBuffer,       // buffer(0)  (b, h, dh)
    pub dt: &'a MTLBuffer,      // buffer(1)  (b, h)
    pub decay: &'a MTLBuffer,   // buffer(2)  (b, h)
    pub b: &'a MTLBuffer,       // buffer(3)  (b, g, n)
    pub c: &'a MTLBuffer,       // buffer(4)  (b, g, n)
    pub d: &'a MTLBuffer,       // buffer(5)  (h)
    pub z: &'a MTLBuffer,       // buffer(6)  (b, d)
    pub state: &'a MTLBuffer,   // buffer(7)  (b, h, dh, n)
    pub y: &'a MTLBuffer,       // buffer(8)
    pub next_state: &'a MTLBuffer, // buffer(9)
    pub group_size: i32,        // buffer(10)
    pub state_size: i32,        // buffer(11)
    pub x_strides: [usize; 3],  // buffer(12)
    pub dt_strides: [usize; 2], // buffer(13)
    pub cb_strides: [usize; 3], // buffer(14)
    pub state_strides: [usize; 4], // buffer(15)
    pub b_size: usize,
    pub h_size: usize,
    pub dh_size: usize,
}

impl SSDUpdateKernel {
    pub fn new(context: &MTLContext, data_type: KernelDataType) -> Result<Self, SSMKernelError> {
        let fn_name = format!("ssd_update_kernel_{}", fn_suffix(data_type));
        let pipeline = context.compute_pipeline_state(&fn_name, None).map_err(SSMKernelError::MetalError)?;
        Ok(Self { pipeline })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: SSDUpdateArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.dt), 0);
        compute_encoder.set_buffer(2, Some(args.decay), 0);
        compute_encoder.set_buffer(3, Some(args.b), 0);
        compute_encoder.set_buffer(4, Some(args.c), 0);
        compute_encoder.set_buffer(5, Some(args.d), 0);
        compute_encoder.set_buffer(6, Some(args.z), 0);
        compute_encoder.set_buffer(7, Some(args.state), 0);
        compute_encoder.set_buffer(8, Some(args.y), 0);
        compute_encoder.set_buffer(9, Some(args.next_state), 0);

        compute_encoder.set_bytes(10, size_of::<i32>() as u64, &args.group_size as *const i32 as *const _);
        compute_encoder.set_bytes(11, size_of::<i32>() as u64, &args.state_size as *const i32 as *const _);
        compute_encoder.set_bytes(12, (3 * size_of::<usize>()) as u64, args.x_strides.as_ptr() as *const _);
        compute_encoder.set_bytes(13, (2 * size_of::<usize>()) as u64, args.dt_strides.as_ptr() as *const _);
        compute_encoder.set_bytes(14, (3 * size_of::<usize>()) as u64, args.cb_strides.as_ptr() as *const _);
        compute_encoder.set_bytes(15, (4 * size_of::<usize>()) as u64, args.state_strides.as_ptr() as *const _);

        let threads_per_threadgroup = MTLSize { width: 32, height: 32, depth: 1 };
        let total_threads = MTLSize { width: args.b_size as u64, height: args.h_size as u64, depth: args.dh_size as u64 };
        compute_encoder.dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
