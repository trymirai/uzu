use thiserror::Error;

use crate::backends::metal::{
    ComputeEncoderSetValue, KernelDataType, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLContext,
    MTLError, MTLSize, ProtocolObject, Retained,
};

pub struct RopeKernel {
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
    pub qkv_buffer: &'a ProtocolObject<dyn MTLBuffer>,     // buffer(0)
    pub cosines_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(1)
    pub sines_buffer: &'a ProtocolObject<dyn MTLBuffer>,   // buffer(2)
    pub token_positions_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(3)
    pub token_positions_offset: usize,                     // byte offset into token_positions_buffer
    pub rotated_queries_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(4)
    pub rotated_keys_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(5)
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
        let function_name = format!("applyRope_{}", data_type.function_name_suffix());

        let pipeline_state = context.compute_pipeline_state(&function_name, None).map_err(|e| {
            eprintln!("Failed to create pipeline state for {}: {:?}", function_name, e);
            RopeError::MetalError(e)
        })?;

        Ok(Self {
            pipeline_state,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: RopeKernelArguments,
    ) {
        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);

        compute_encoder.set_buffer(Some(args.qkv_buffer), 0, 0);
        compute_encoder.set_buffer(Some(args.cosines_buffer), 0, 1);
        compute_encoder.set_buffer(Some(args.sines_buffer), 0, 2);
        compute_encoder.set_buffer(Some(args.token_positions_buffer), args.token_positions_offset as usize, 3);
        compute_encoder.set_buffer(Some(args.rotated_queries_buffer), 0, 4);
        compute_encoder.set_buffer(Some(args.rotated_keys_buffer), 0, 5);

        // Set constants
        let head_dim = args.head_dim as u32;
        let num_heads = args.num_heads as u32;
        let num_groups = args.num_groups as u32;
        let suffix_length = args.suffix_length as u32;
        let max_sequence_length = args.max_sequence_length as u32;

        compute_encoder.set_value(&head_dim, 6);
        compute_encoder.set_value(&num_heads, 7);
        compute_encoder.set_value(&num_groups, 8);
        compute_encoder.set_value(&suffix_length, 9);
        compute_encoder.set_value(&max_sequence_length, 10);

        let threads_per_threadgroup = MTLSize {
            width: 1,
            height: 1,
            depth: 32,
        };
        let total_threads = MTLSize {
            width: num_heads as usize,
            height: suffix_length as usize,
            depth: head_dim as usize,
        };

        compute_encoder.dispatch_threads(total_threads, threads_per_threadgroup);
    }
}
