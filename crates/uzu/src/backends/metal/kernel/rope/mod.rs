use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};
use thiserror::Error;

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

pub struct RopeKernel {
    pipeline_state: MTLComputePipelineState,
}

#[derive(Debug, Error)]
pub enum RopeError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
}

#[allow(dead_code)]
pub struct RopeKernelArguments<'a> {
    pub qkv_buffer: &'a MTLBuffer,     // buffer(0)
    pub cosines_buffer: &'a MTLBuffer, // buffer(1)
    pub sines_buffer: &'a MTLBuffer,   // buffer(2)
    pub token_positions_buffer: &'a MTLBuffer, // buffer(3)
    pub token_positions_offset: usize, // byte offset into token_positions_buffer
    pub rotated_queries_buffer: &'a MTLBuffer, // buffer(4)
    pub rotated_keys_buffer: &'a MTLBuffer, // buffer(5)
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub suffix_length: usize,
    pub max_sequence_length: usize,
}

impl RopeKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, RopeError> {
        let function_name =
            format!("applyRope_{}", data_type.function_name_suffix());

        let (pipeline_state, _argument_names) = context
            .compute_pipeline_state_with_reflection(&function_name, None)
            .map_err(|e| {
                eprintln!(
                    "Failed to create pipeline state for {}: {:?}",
                    function_name, e
                );
                RopeError::MetalError(e)
            })?;

        Ok(Self {
            pipeline_state,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: RopeKernelArguments,
    ) {
        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);

        compute_encoder.set_buffer(0, Some(args.qkv_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.cosines_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.sines_buffer), 0);
        compute_encoder.set_buffer(
            3,
            Some(args.token_positions_buffer),
            args.token_positions_offset as u64,
        );
        compute_encoder.set_buffer(4, Some(args.rotated_queries_buffer), 0);
        compute_encoder.set_buffer(5, Some(args.rotated_keys_buffer), 0);

        // Set constants
        let head_dim = args.head_dim as u32;
        let num_heads = args.num_heads as u32;
        let num_groups = args.num_groups as u32;
        let suffix_length = args.suffix_length as u32;
        let max_sequence_length = args.max_sequence_length as u32;

        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &num_heads as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &num_groups as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &suffix_length as *const u32 as *const _,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &max_sequence_length as *const u32 as *const _,
        );

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: num_heads as u64,
            height: suffix_length as u64,
            depth: head_dim as u64,
        };

        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
    }
}
