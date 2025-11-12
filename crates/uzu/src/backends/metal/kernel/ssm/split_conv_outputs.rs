use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef, MTLSize};

use crate::backends::metal::{KernelDataType, MTLContext};

use super::{SSMKernelError, fn_suffix};

pub struct SplitConvOutputsKernel {
    pipeline: metal::ComputePipelineState,
}

pub struct SplitConvOutputsArguments<'a> {
    pub conv_input: &'a MTLBuffer, // buffer(0) [suffix, conv_dim]
    pub x_out: &'a MTLBuffer,      // buffer(1) [suffix, inner_dim]
    pub b_out: &'a MTLBuffer,      // buffer(2) [suffix, proj_dim]
    pub c_out: &'a MTLBuffer,      // buffer(3) [suffix, proj_dim]
    pub conv_dim: usize,
    pub inner_dim: usize,
    pub proj_dim: usize,
    pub suffix_length: usize,
}

impl SplitConvOutputsKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name =
            format!("ssm_split_conv_outputs_kernel_{}", fn_suffix(data_type));
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
        args: SplitConvOutputsArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(0, Some(args.conv_input), 0);
        compute_encoder.set_buffer(1, Some(args.x_out), 0);
        compute_encoder.set_buffer(2, Some(args.b_out), 0);
        compute_encoder.set_buffer(3, Some(args.c_out), 0);

        let conv_dim = args.conv_dim as i32;
        let inner_dim = args.inner_dim as i32;
        let proj_dim = args.proj_dim as i32;
        let suffix = args.suffix_length as u64;
        let cols = args.conv_dim as u64;

        compute_encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &conv_dim as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &inner_dim as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &proj_dim as *const i32 as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: suffix,
            height: cols,
            depth: 1,
        };

        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
