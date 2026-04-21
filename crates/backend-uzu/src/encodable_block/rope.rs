//! Rope (Rotary Position Embedding) encodable.

use std::ops::{Deref, DerefMut};

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
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

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        use_rope: bool,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let token_positions = state.array(ArrayId::TokenPositions);
        let qkv = state.array(ArrayId::QKV);
        let cosines = state.array(ArrayId::RopeCosines(self.rope_type));
        let sines = state.array(ArrayId::RopeSines(self.rope_type));
        let rotated_queries = state.array(ArrayId::RotatedQueries);
        let rotated_keys = state.array(ArrayId::RotatedKeys);

        let suffix_length = qkv.shape()[0];

        let rope_max_seq_len = cosines.shape()[0];
        let rope_dim = if use_rope {
            cosines.shape()[1]
        } else {
            0
        };

        let num_heads = rotated_queries.shape()[0];
        let head_dim = rotated_queries.shape()[2];

        let num_groups = rotated_keys.shape()[0];

        self.kernel.encode(
            qkv.buffer().borrow().deref(),
            cosines.buffer().borrow().deref(),
            sines.buffer().borrow().deref(),
            (token_positions.buffer().borrow().deref(), token_positions.offset()),
            rotated_queries.buffer().borrow_mut().deref_mut(),
            rotated_keys.buffer().borrow_mut().deref_mut(),
            head_dim as u32,
            rope_dim as u32,
            num_heads as u32,
            num_groups as u32,
            suffix_length as u32,
            rope_max_seq_len as u32,
            encoder,
        );
        Ok(())
    }
}
