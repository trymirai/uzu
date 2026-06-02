//! Rope (Rotary Position Embedding) encodable.

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, RopeKernel},
    },
    data_type::DataType,
    forward_pass::model_shape::ModelShape,
};

pub struct Rope<B: Backend> {
    kernel: <B::Kernels as Kernels>::RopeKernel,
    data_type: DataType,
    query_only: bool,
}

impl<B: Backend> Rope<B> {
    pub fn new(
        context: &B::Context,
        model_shape: &ModelShape,
        query_only: bool,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            kernel: <B::Kernels as Kernels>::RopeKernel::new(
                context,
                model_shape.data_type,
                model_shape.rope_data_type,
                query_only,
            )?,
            data_type: model_shape.data_type,
            query_only,
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
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Option<Allocation<B>>), B::Error> {
        let mut rotated_queries =
            encoder.allocate_scratch(size_for_shape(&[num_heads, suffix_length, head_dim], self.data_type))?;
        let mut rotated_keys = if self.query_only {
            None
        } else {
            Some(encoder.allocate_scratch(size_for_shape(&[num_groups, suffix_length, head_dim], self.data_type))?)
        };
        self.kernel.encode(
            qkv,
            cosines,
            sines,
            token_positions,
            &mut rotated_queries,
            rotated_keys.as_mut(),
            head_dim as u32,
            rope_dim as u32,
            num_heads as u32,
            (!self.query_only).then_some(num_groups as u32),
            suffix_length as u32,
            rope_max_seq_len as u32,
            encoder,
        );
        Ok((rotated_queries, rotated_keys))
    }
}
