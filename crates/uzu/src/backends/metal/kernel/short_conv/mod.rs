use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

#[derive(Debug, thiserror::Error)]
pub enum ShortConvKernelError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

fn fn_suffix(dt: KernelDataType) -> &'static str {
    dt.function_name_suffix()
}

pub struct ShortConvKernel {
    prefill_pipeline: MTLComputePipelineState,
    decode_pipeline: MTLComputePipelineState,
}

pub struct ShortConvPrefillArguments<'a> {
    pub in_proj: &'a MTLBuffer,
    pub w: &'a MTLBuffer,
    pub b: Option<&'a MTLBuffer>,
    pub state_in: &'a MTLBuffer,
    pub out: &'a MTLBuffer,
    pub state_out: &'a MTLBuffer,
    pub suffix_len: usize,
    pub kernel_size: i32,
    pub in_proj_stride: usize,
    pub state_stride: usize,
    pub model_dim: usize,
}

pub struct ShortConvDecodeArguments<'a> {
    pub in_proj: &'a MTLBuffer,
    pub w: &'a MTLBuffer,
    pub b: Option<&'a MTLBuffer>,
    pub state: &'a MTLBuffer,
    pub out: &'a MTLBuffer,
    pub next_state: &'a MTLBuffer,
    pub suffix_len: usize,
    pub kernel_size: i32,
    pub in_proj_stride: usize,
    pub state_stride: usize,
    pub model_dim: usize,
}

impl ShortConvKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, ShortConvKernelError> {
        let prefill_name =
            format!("short_conv_prefill_kernel_{}", fn_suffix(data_type));
        let decode_name =
            format!("short_conv_decode_kernel_{}", fn_suffix(data_type));

        let prefill_pipeline = context
            .compute_pipeline_state(&prefill_name, None)
            .map_err(ShortConvKernelError::MetalError)?;
        let decode_pipeline = context
            .compute_pipeline_state(&decode_name, None)
            .map_err(ShortConvKernelError::MetalError)?;

        Ok(Self {
            prefill_pipeline,
            decode_pipeline,
        })
    }

    pub fn encode_prefill(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: ShortConvPrefillArguments,
    ) -> Result<(), ShortConvKernelError> {
        if args.model_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }

        compute_encoder.set_compute_pipeline_state(&self.prefill_pipeline);
        compute_encoder.set_buffer(0, Some(args.in_proj), 0);
        compute_encoder.set_buffer(1, Some(args.w), 0);
        compute_encoder.set_buffer(2, args.b.map(|v| &**v), 0);
        compute_encoder.set_buffer(3, Some(args.state_in), 0);
        compute_encoder.set_buffer(4, Some(args.out), 0);
        compute_encoder.set_buffer(5, Some(args.state_out), 0);
        compute_encoder.set_bytes(
            6,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &args.kernel_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<usize>() as u64,
            &args.in_proj_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<usize>() as u64,
            &args.state_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &(args.model_dim as u32) as *const u32 as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: args.suffix_len as u64,
            height: args.model_dim as u64,
            depth: 1,
        };

        compute_encoder.dispatch_threads(threadgroups, threads_per_threadgroup);

        Ok(())
    }

    pub fn encode_decode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: ShortConvDecodeArguments,
    ) -> Result<(), ShortConvKernelError> {
        if args.model_dim == 0 || args.suffix_len == 0 {
            return Ok(());
        }

        compute_encoder.set_compute_pipeline_state(&self.decode_pipeline);
        compute_encoder.set_buffer(0, Some(args.in_proj), 0);
        compute_encoder.set_buffer(1, Some(args.w), 0);
        compute_encoder.set_buffer(2, args.b.map(|v| &**v), 0);
        compute_encoder.set_buffer(3, Some(args.state), 0);
        compute_encoder.set_buffer(4, Some(args.out), 0);
        compute_encoder.set_buffer(5, Some(args.next_state), 0);
        compute_encoder.set_bytes(
            6,
            size_of::<usize>() as u64,
            &args.suffix_len as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<i32>() as u64,
            &args.kernel_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<usize>() as u64,
            &args.in_proj_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<usize>() as u64,
            &args.state_stride as *const usize as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &(args.model_dim as u32) as *const u32 as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: args.suffix_len as u64,
            height: args.model_dim as u64,
            depth: 1,
        };

        compute_encoder.dispatch_threads(threadgroups, threads_per_threadgroup);

        Ok(())
    }
}
