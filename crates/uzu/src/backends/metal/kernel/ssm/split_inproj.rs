use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef, MTLSize};

use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{KernelDataType, MTLContext};

pub struct SplitInProjKernel {
    pipeline: metal::ComputePipelineState,
}

pub struct SplitInProjArguments<'a> {
    pub input: &'a MTLBuffer, // buffer(0) [suffix, total_dim]
    pub conv_out: &'a MTLBuffer, // buffer(1) [suffix, conv_dim]
    pub z_out: &'a MTLBuffer, // buffer(2) [suffix, inner_dim]
    pub dt_out: &'a MTLBuffer, // buffer(3) [suffix, num_heads]
    pub z_bias: &'a MTLBuffer, // buffer(4) [inner_dim]
    pub total_dim: usize,
    pub conv_dim: usize,
    pub inner_dim: usize,
    pub num_heads: usize,
    pub suffix_length: usize,
}

impl SplitInProjKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name =
            format!("ssm_split_inproj_kernel_{}", fn_suffix(data_type));
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
        args: SplitInProjArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(0, Some(args.input), 0);
        compute_encoder.set_buffer(1, Some(args.conv_out), 0);
        compute_encoder.set_buffer(2, Some(args.z_out), 0);
        compute_encoder.set_buffer(3, Some(args.dt_out), 0);
        compute_encoder.set_buffer(4, Some(args.z_bias), 0);

        let total_dim = args.total_dim as i32;
        let conv_dim = args.conv_dim as i32;
        let inner_dim = args.inner_dim as i32;
        let num_heads = args.num_heads as i32;
        let suffix = args.suffix_length as u64;
        let cols = args.total_dim as u64;

        compute_encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &total_dim as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &conv_dim as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            std::mem::size_of::<i32>() as u64,
            &inner_dim as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            std::mem::size_of::<i32>() as u64,
            &num_heads as *const i32 as *const _,
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
