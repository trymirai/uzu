//! Rope (Rotary Position Embedding) encodable.

use super::{EncodableBlock, Metal};
use crate::{
    backends::metal::{
        KernelDataType, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLContext, ProtocolObject,
        Retained,
        kernel::rope::{RopeError, RopeKernel, RopeKernelArguments},
    },
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState, RopeType},
};

pub struct Rope {
    kernel: RopeKernel,
    rope_type: RopeType,
}

impl Rope {
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

impl EncodableBlock<Metal> for Rope {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        parameters: &EncodingParameters<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        let compute_encoder =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.encode_with_shared_encoder(state, parameters, &compute_encoder);
        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<Metal>,
        _parameters: &EncodingParameters<Metal>,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
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

            let cos_binding = state.arrays(&[ArrayId::RopeCosines(self.rope_type)]);
            let cos_array = cos_binding[0].borrow();
            let cos_shape = cos_array.shape();
            let rope_max_seq_len = cos_shape[0];

            (suffix_length, num_heads, head_dim, num_groups, rope_max_seq_len)
        };

        let qkv_buffer_binding = state.arrays(&[ArrayId::QKV]);
        let qkv = qkv_buffer_binding[0].borrow_mut();

        let token_positions_binding = state.arrays(&[ArrayId::TokenPositions]);
        let token_positions = token_positions_binding[0].borrow_mut();

        let query_buffer_binding = state.arrays(&[ArrayId::RotatedQueries]);
        let rotated_queries = query_buffer_binding[0].borrow_mut();

        let rotated_keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
        let rotated_keys = rotated_keys_binding[0].borrow_mut();

        let cos_buffer_binding = state.arrays(&[ArrayId::RopeCosines(self.rope_type)]);
        let rope_cosines = cos_buffer_binding[0].borrow_mut();

        let sin_buffer_binding = state.arrays(&[ArrayId::RopeSines(self.rope_type)]);
        let rope_sines = sin_buffer_binding[0].borrow_mut();

        let token_positions_offset = token_positions.offset();

        self.kernel.encode(
            compute_encoder,
            RopeKernelArguments {
                qkv_buffer: &qkv.buffer(),
                cosines_buffer: &rope_cosines.buffer(),
                sines_buffer: &rope_sines.buffer(),
                token_positions_buffer: &token_positions.buffer(),
                token_positions_offset,
                rotated_queries_buffer: &rotated_queries.buffer(),
                rotated_keys_buffer: &rotated_keys.buffer(),
                head_dim,
                num_heads,
                num_groups,
                suffix_length,
                max_sequence_length: rope_max_seq_len,
            },
        );
    }
}
