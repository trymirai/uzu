use metal::{Buffer as MTLBuffer, ComputeCommandEncoderRef, MTLSize};

use crate::backends::metal::{KernelDataType, MTLContext};

use super::{SSMKernelError, fn_suffix};

pub struct ActivationKernel {
    pipeline: metal::ComputePipelineState,
}

pub enum ActivationType {
    Identity = 0,
    Silu = 1,
    Gelu = 2,
}

pub struct ActivationArguments<'a> {
    pub data: &'a MTLBuffer, // buffer(0) [suffix, row_stride]
    pub row_stride: usize,
    pub suffix_length: usize,
    pub activation: ActivationType,
}

impl ActivationKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("ssm_activation_kernel_{}", fn_suffix(data_type));
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
        args: ActivationArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(0, Some(args.data), 0);

        let row_stride = args.row_stride as i32;
        let suffix = args.suffix_length as i32;
        let activation = args.activation as i32;

        compute_encoder.set_bytes(
            1,
            std::mem::size_of::<i32>() as u64,
            &activation as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &row_stride as *const i32 as *const _,
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
            height: args.row_stride as u64,
            depth: 1,
        };

        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
