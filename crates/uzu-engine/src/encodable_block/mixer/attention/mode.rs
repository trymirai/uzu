use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{AttentionArguments, AttentionKernel, AttentionPrepareKernel, SigmoidGateKernel},
    },
    encodable_block::{
        batch_topology::BatchTopology,
        linear::Linear,
        mixer::{
            MixerState,
            attention::{Attention, KVCacheView, qkv_norm::QKVNorm, rope::PrecalculatedRoPE, state::AttentionState},
        },
    },
    utils::maybe_mut::MaybeMut,
};

pub(super) struct LinearProjection<B: Backend> {
    pub(super) lin: Box<dyn Linear<B>>,
    pub(super) norm: Option<QKVNorm<B>>,
}

impl<B: Backend> LinearProjection<B> {
    fn project(
        &self,
        hidden: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut projected = self.lin.encode(hidden, batch_dim, command_buffer)?;
        if let Some(norm) = &self.norm {
            norm.encode(&mut projected, batch_dim, command_buffer)?;
        }
        Ok(projected)
    }
}

impl<B: Backend> Attention<B> {
    pub(super) fn attend(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let qkvg = self.projection.project(hidden, batch_dim.size(), command_buffer)?;

        let mut attention_output = match state {
            Some(MaybeMut::Mut(state)) => {
                let prefix_len = state.view().prefix_len();
                let queries = self.prepare_kv_and_queries(
                    &qkvg,
                    state.keys.as_mut(),
                    state.values.as_mut(),
                    prefix_len,
                    self.num_q_heads,
                    precalculated_rope,
                    batch_dim.size(),
                    command_buffer,
                )?;
                self.run_core(&queries, batch_dim, state, command_buffer)?
            },
            Some(MaybeMut::Const(state)) => {
                // KV sharing: QKVG contains queries and an optional gate only.
                let queries = self.prepare_queries(&qkvg, precalculated_rope, batch_dim.size(), command_buffer)?;
                self.run_core(&queries, batch_dim, state, command_buffer)?
            },
            None => {
                let Some(num_kv_heads) = self.num_kv_heads else {
                    panic!("stateless attention doesn't support query-only projection");
                };
                assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

                let mut keys = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)?;
                let mut values = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)?;

                let queries = self.prepare_kv_and_queries(
                    &qkvg,
                    &mut keys,
                    &mut values,
                    0,
                    self.num_q_heads,
                    precalculated_rope,
                    batch_dim.size(),
                    command_buffer,
                )?;

                let cache = self.ring_capacity.map_or_else(|| KVCacheView::full(0), |_| KVCacheView::ring(0, 0));

                self.kernel.encode(
                    AttentionArguments {
                        queries: &queries,
                        keys: &keys,
                        values: &values,
                        suffix_length: batch_dim.size(),
                        trie: None::<&B::ConstantBuffer>,
                        sinks: self.sinks.as_ref(),
                        cache,
                    },
                    command_buffer,
                )?
            },
        };

        if let Some(gate_kernel) = &self.gate_kernel {
            let gate_dim = self.num_q_heads * self.head_dim;
            let gate_offset = self.projection_dim - gate_dim;
            gate_kernel.encode(
                qkvg.subrange(gate_offset as usize * self.data_type.size_in_bytes()..),
                &mut attention_output,
                gate_dim,
                batch_dim.size(),
                self.projection_dim,
                command_buffer,
            );
        }
        self.out_projection.encode(attention_output, batch_dim.size(), command_buffer)
    }

    pub fn append_projected_kv(
        &self,
        mut key_value: impl BufferMut<Backend = B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: u32,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        if let Some(norm) = &self.projection.norm {
            norm.encode_key_value(key_value.reborrow(), batch_dim, command_buffer)?;
        }
        let prefix_len = state.view().prefix_len();
        self.prepare_kv_and_queries(
            key_value.as_ref(),
            state.keys.as_mut(),
            state.values.as_mut(),
            prefix_len,
            0,
            Some(precalculated_rope),
            batch_dim,
            command_buffer,
        )?;
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), command_buffer)?;
        Ok(())
    }

    fn run_core(
        &self,
        queries: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        state: &AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let trie = if batch_dim.is_flat() {
            None
        } else {
            Some(command_buffer.allocate_constant_from_slice(batch_dim.nodes())?)
        };

        self.kernel.encode(
            AttentionArguments {
                queries,
                keys: state.keys.as_ref(),
                values: state.values.as_ref(),
                suffix_length: batch_dim.size(),
                trie: trie.as_ref(),
                sinks: self.sinks.as_ref(),
                cache: state.view(),
            },
            command_buffer,
        )
    }

    fn prepare_kv_and_queries(
        &self,
        input: impl BufferRef<Backend = B>,
        keys: impl BufferMut<Backend = B>,
        values: impl BufferMut<Backend = B>,
        kv_token_offset: u32,
        num_q_heads: u32,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let num_kv_heads = self.num_kv_heads.expect("KV prepare requires KV heads");
        // Appended KV is tightly packed; attention projections may have a trailing gate segment.
        let input_row_stride = if num_q_heads == 0 {
            2 * num_kv_heads * self.head_dim
        } else {
            self.projection_dim
        };
        let mut queries = if num_q_heads == 0 {
            command_buffer.allocate_scratch(self.data_type.size_in_bytes())?
        } else {
            command_buffer.allocate_scratch_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type)?
        };
        self.prepare.encode(
            input,
            &mut queries,
            Some(keys),
            Some(values),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
            num_q_heads,
            Some(num_kv_heads),
            self.head_dim,
            precalculated_rope.map(|precalculated_rope| precalculated_rope.dim),
            Some(kv_token_offset),
            input_row_stride,
            batch_dim,
            command_buffer,
        );
        Ok(queries)
    }

    fn prepare_queries(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut queries =
            command_buffer.allocate_scratch_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type)?;
        self.prepare.encode(
            qkvg,
            &mut queries,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            precalculated_rope.map(|rope| &rope.cosines),
            precalculated_rope.map(|rope| &rope.sines),
            self.num_q_heads,
            None,
            self.head_dim,
            precalculated_rope.map(|rope| rope.dim),
            None,
            self.projection_dim,
            batch_dim,
            command_buffer,
        );
        Ok(queries)
    }
}
