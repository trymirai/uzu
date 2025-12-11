use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use thiserror::Error;

use crate::{
    Array,
    backends::metal::{
        KernelDataType, MTLContext, MTLError,
        forward_pass::{
            ArrayId, ForwardPassState, RopeType,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
        metal_extensions::ComputeEncoderConditional,
    },
};

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

pub struct RopeKernelEncodable {
    kernel: RopeKernel,
    rope_type: RopeType,
}

impl RopeKernelEncodable {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        rope_type: RopeType,
    ) -> Result<Self, RopeError> {
        let kernel = RopeKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            rope_type,
        })
    }
}

impl EncodableWithState for RopeKernelEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let compute_encoder = mtl_command_buffer.new_compute_command_encoder();

        compute_encoder.condition(
            parameters.predicate_ref(),
            || {
                self.encode_with_encoder_impl(state, compute_encoder);
            },
            None::<fn()>,
        );

        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
        _parameters: &EncodingParameters,
    ) {
        self.encode_with_encoder_impl(state, encoder);
    }
}

impl RopeKernelEncodable {
    fn encode_with_encoder_impl(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
    ) {
        let (suffix_length, num_heads, head_dim, num_groups, rope_max_seq_len) = {
            let qkv_binding = state.arrays(&[ArrayId::QKV]);
            let qkv_array = qkv_binding[0].borrow();
            let suffix_length = qkv_array.shape()[0];

            let queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
            let queries_array = queries_binding[0].borrow();
            let num_heads = queries_array.shape()[0];
            let head_dim = queries_array.shape()[2];

            let keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
            let keys_array = keys_binding[0].borrow();
            let num_groups = keys_array.shape()[0];

            let cos_binding =
                state.arrays(&[ArrayId::RopeCosines(self.rope_type)]);
            let cos_array = cos_binding[0].borrow();
            let cos_shape = cos_array.shape();
            let rope_max_seq_len = cos_shape[0];

            (suffix_length, num_heads, head_dim, num_groups, rope_max_seq_len)
        };

        let qkv_buffer_binding = state.arrays(&[ArrayId::QKV]);
        let mut qkv = qkv_buffer_binding[0].borrow_mut();

        let token_positions_binding = state.arrays(&[ArrayId::TokenPositions]);
        let mut token_positions = token_positions_binding[0].borrow_mut();

        let query_buffer_binding = state.arrays(&[ArrayId::RotatedQueries]);
        let mut rotated_queries = query_buffer_binding[0].borrow_mut();

        let rotated_keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
        let mut rotated_keys = rotated_keys_binding[0].borrow_mut();

        let cos_buffer_binding =
            state.arrays(&[ArrayId::RopeCosines(self.rope_type)]);
        let mut rope_cosines = cos_buffer_binding[0].borrow_mut();

        let sin_buffer_binding =
            state.arrays(&[ArrayId::RopeSines(self.rope_type)]);
        let mut rope_sines = sin_buffer_binding[0].borrow_mut();

        let token_positions_offset = token_positions.buffer_offset();

        self.kernel.encode(
            encoder,
            RopeKernelArguments {
                qkv_buffer: unsafe { &qkv.mtl_buffer() },
                cosines_buffer: unsafe { &rope_cosines.mtl_buffer() },
                sines_buffer: unsafe { &rope_sines.mtl_buffer() },
                token_positions_buffer: unsafe {
                    &token_positions.mtl_buffer()
                },
                token_positions_offset,
                rotated_queries_buffer: unsafe {
                    &rotated_queries.mtl_buffer()
                },
                rotated_keys_buffer: unsafe { &rotated_keys.mtl_buffer() },
                head_dim,
                num_heads,
                num_groups,
                suffix_length,
                max_sequence_length: rope_max_seq_len,
            },
        );
    }
}
