//! Rope (Rotary Position Embedding) encodable.

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{Kernels, RopeKernel},
    },
    forward_pass::state::{ArrayId, ForwardPassState, RopeType},
};

pub struct Rope<B: Backend> {
    kernel: <B::Kernels as Kernels>::RopeKernel,
    rope_type: RopeType,
}

impl<B: Backend> Rope<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        rope_type: RopeType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            kernel: <B::Kernels as Kernels>::RopeKernel::new(context, data_type)?,
            rope_type,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for Rope<B> {
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
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
            qkv.buffer(),
            rope_cosines.buffer(),
            rope_sines.buffer(),
            (token_positions.buffer(), token_positions_offset),
            rotated_queries.buffer(),
            rotated_keys.buffer(),
            head_dim as u32,
            num_heads as u32,
            num_groups as u32,
            suffix_length as u32,
            rope_max_seq_len as u32,
            encoder,
        )
    }
}
