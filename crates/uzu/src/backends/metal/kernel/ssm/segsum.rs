use std::mem::size_of;

use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef, MTLSize};

use crate::backends::metal::{KernelDataType, MTLContext};

use super::{SSMKernelError, fn_suffix};

pub struct Cumsum1DKernel {
    pipeline: metal::ComputePipelineState,
}

pub struct Cumsum1DArguments<'a> {
    pub x: &'a MTLBuffer,     // buffer(0) - (outer, length)
    pub s: &'a MTLBuffer,     // buffer(1) - (outer, length)
    pub length: usize,        // buffer(2)
    pub outer_size: usize,    // buffer(3)
}

impl Cumsum1DKernel {
    pub fn new(context: &MTLContext, data_type: KernelDataType) -> Result<Self, SSMKernelError> {
        let fn_name = format!("cumsum_1d_kernel_{}", fn_suffix(data_type));
        let pipeline = context
            .compute_pipeline_state(&fn_name, None)
            .map_err(SSMKernelError::MetalError)?;
        Ok(Self { pipeline })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: Cumsum1DArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.s), 0);

        compute_encoder.set_bytes(2, size_of::<usize>() as u64, &args.length as *const usize as *const _);
        compute_encoder.set_bytes(3, size_of::<usize>() as u64, &args.outer_size as *const usize as *const _);

        let threads_per_grid = MTLSize { width: args.outer_size as u64, height: 1, depth: 1 };
        compute_encoder.dispatch_threads(threads_per_grid, MTLSize { width: 1, height: 1, depth: 1 });
        Ok(())
    }
}

pub struct SegsumFromCumsumKernel {
    pipeline: metal::ComputePipelineState,
}

pub struct SegsumFromCumsumArguments<'a> {
    pub s: &'a MTLBuffer,      // buffer(0) - (outer, length)
    pub y: &'a MTLBuffer,      // buffer(1) - (outer, length, length)
    pub length: usize,         // buffer(2)
    pub outer_size: usize,     // buffer(3)
}

impl SegsumFromCumsumKernel {
    pub fn new(context: &MTLContext, data_type: KernelDataType) -> Result<Self, SSMKernelError> {
        let fn_name = format!("segsum_from_cumsum_kernel_{}", fn_suffix(data_type));
        let pipeline = context
            .compute_pipeline_state(&fn_name, None)
            .map_err(SSMKernelError::MetalError)?;
        Ok(Self { pipeline })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: SegsumFromCumsumArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(0, Some(args.s), 0);
        compute_encoder.set_buffer(1, Some(args.y), 0);

        compute_encoder.set_bytes(2, size_of::<usize>() as u64, &args.length as *const usize as *const _);
        compute_encoder.set_bytes(3, size_of::<usize>() as u64, &args.outer_size as *const usize as *const _);

        let threads_per_grid = MTLSize { width: args.outer_size as u64, height: args.length as u64, depth: 1 };
        compute_encoder.dispatch_threads(threads_per_grid, MTLSize { width: 1, height: 1, depth: 1 });
        Ok(())
    }
}



