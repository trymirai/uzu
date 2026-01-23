use std::{ffi::c_void, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{
    ComputePipelineState, KernelDataType, MTLBuffer,
    MTLContext, MTLSize, ProtocolObject,
};

pub struct SplitInProjKernel {
    pipeline: ComputePipelineState,
}

pub struct SplitInProjArguments<'a> {
    pub input: &'a ProtocolObject<dyn MTLBuffer>, // buffer(0) [suffix, total_dim]
    pub conv_out: &'a ProtocolObject<dyn MTLBuffer>, // buffer(1) [suffix, conv_dim]
    pub z_out: &'a ProtocolObject<dyn MTLBuffer>, // buffer(2) [suffix, inner_dim]
    pub dt_out: &'a ProtocolObject<dyn MTLBuffer>, // buffer(3) [suffix, num_heads]
    pub z_bias: &'a ProtocolObject<dyn MTLBuffer>, // buffer(4) [inner_dim]
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
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: SplitInProjArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(Some(args.input), 0, 0);
        compute_encoder.set_buffer(Some(args.conv_out), 0, 1);
        compute_encoder.set_buffer(Some(args.z_out), 0, 2);
        compute_encoder.set_buffer(Some(args.dt_out), 0, 3);
        compute_encoder.set_buffer(Some(args.z_bias), 0, 4);

        let total_dim = args.total_dim as i32;
        let conv_dim = args.conv_dim as i32;
        let inner_dim = args.inner_dim as i32;
        let num_heads = args.num_heads as i32;
        let suffix = args.suffix_length;
        let cols = args.total_dim;

        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&total_dim as *const i32 as *mut c_void).unwrap(),
                std::mem::size_of::<i32>(),
                5,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&conv_dim as *const i32 as *mut c_void).unwrap(),
                std::mem::size_of::<i32>(),
                6,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&inner_dim as *const i32 as *mut c_void).unwrap(),
                std::mem::size_of::<i32>(),
                7,
            );
        }
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&num_heads as *const i32 as *mut c_void).unwrap(),
                std::mem::size_of::<i32>(),
                8,
            );
        }

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
