//! Rope (Rotary Position Embedding) encodable.

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, RopeKernel},
    },
    forward_pass::state::RopeType,
};

pub struct Rope<B: Backend> {
    kernel: <B::Kernels as Kernels>::RopeKernel,
    rope_type: RopeType,
    data_type: DataType,
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
            data_type,
        })
    }

    pub fn encode(
        &self,
        qkv: &Allocation<B>,
        token_positions: &Allocation<B>,
        cosines: &Allocation<B>,
        sines: &Allocation<B>,
        suffix_length: usize,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
        rope_max_seq_len: usize,
        rope_dim: usize,
        use_rope: bool,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), B::Error> {
        let applied_rope_dim = if use_rope {
            assert!(rope_dim > 0, "enabled RoPE dimension must be greater than zero");
            rope_dim
        } else {
            0
        };
        let mut rotated_queries =
            encoder.allocate_scratch(size_for_shape(&[num_heads, suffix_length, head_dim], self.data_type))?;
        let mut rotated_keys =
            encoder.allocate_scratch(size_for_shape(&[num_groups, suffix_length, head_dim], self.data_type))?;
        self.kernel.encode(
            qkv,
            cosines,
            sines,
            token_positions,
            &mut rotated_queries,
            &mut rotated_keys,
            head_dim as u32,
            applied_rope_dim as u32,
            num_heads as u32,
            num_groups as u32,
            suffix_length as u32,
            rope_max_seq_len as u32,
            encoder,
        );
        Ok((rotated_queries, rotated_keys))
    }

    pub fn rope_type(&self) -> RopeType {
        self.rope_type
    }
}
