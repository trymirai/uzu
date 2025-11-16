use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext};

use super::{SSMKernelError, fn_suffix};

const SSD_PREFILL_THREADGROUP_WIDTH: u64 = 64;

pub struct SSDPrefillKernel {
    pipeline: MTLComputePipelineState,
}

pub struct SSDPrefillArguments<'a> {
    pub x: &'a MTLBuffer,     // buffer(0) [suffix, h, dh]
    pub dt: &'a MTLBuffer,    // buffer(1) [suffix, h]
    pub decay: &'a MTLBuffer, // buffer(2) [suffix, h]
    pub b: &'a MTLBuffer,     // buffer(3) [suffix, g, n]
    pub c: &'a MTLBuffer,     // buffer(4) [suffix, g, n]
    pub d: &'a MTLBuffer,     // buffer(5) [h]
    pub z: &'a MTLBuffer,     // buffer(6) [suffix, h, dh]
    pub state: &'a MTLBuffer, // buffer(7) [h, dh, n]
    pub y: &'a MTLBuffer,     // buffer(8) [suffix, h, dh]
    pub suffix_len: usize,
    pub group_size: i32,
    pub state_size: i32,
    pub x_strides: [usize; 3],
    pub dt_strides: [usize; 2],
    pub cb_strides: [usize; 3],
    pub state_strides: [usize; 3],
    pub channels: usize,
    pub head_dim: usize,
}

impl SSDPrefillKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("ssd_prefill_kernel_{}", fn_suffix(data_type));
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
        args: SSDPrefillArguments,
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

        compute_encoder.set_bytes(
            9,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<i32>() as u64,
            &args.group_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<i32>() as u64,
            &args.state_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            12,
            (3 * size_of::<usize>()) as u64,
            args.x_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            13,
            (2 * size_of::<usize>()) as u64,
            args.dt_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            14,
            (3 * size_of::<usize>()) as u64,
            args.cb_strides.as_ptr() as *const _,
        );
        compute_encoder.set_bytes(
            15,
            (3 * size_of::<usize>()) as u64,
            args.state_strides.as_ptr() as *const _,
        );
        let channels = args.channels as u32;
        let head_dim = args.head_dim as u32;
        compute_encoder.set_bytes(
            16,
            size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            17,
            size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: SSD_PREFILL_THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: args.channels as u64 * args.head_dim as u64
                * SSD_PREFILL_THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        compute_encoder.dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
