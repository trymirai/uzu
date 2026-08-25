use crate::{
    backends::common::{
        Allocation, Backend, BufferArgMut, Encoder,
        gpu_types::trie::TrieNode,
        kernel::{AttentionArguments, AttentionKernel, AttentionPrepareKernel, SigmoidGateKernel},
    },
    encodable_block::{
        batch_topology::BatchTopology,
        linear::Linear,
        mixer::{
            MixerState,
            attention::{
                Attention,
                qkv_norm::QKVNorm,
                rope::PrecalculatedRoPE,
                state::{AttentionState, AttentionStateType},
            },
        },
    },
    utils::maybe_mut::MaybeMut,
};

pub struct LinearProjection<B: Backend> {
    pub lin: Box<dyn Linear<B>>,
    pub norm: Option<QKVNorm<B>>,
}

impl<B: Backend> LinearProjection<B> {
    fn project(
        &self,
        hidden: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut projected = self.lin.encode(hidden, batch_dim, encoder)?;
        if let Some(norm) = &self.norm {
            norm.encode(&mut projected, batch_dim, encoder)?;
        }
        Ok(projected)
    }
}

impl<B: Backend> Attention<B> {
    pub fn attend(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<AttentionState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        // If we have gate we must duplicate input (linear does hadamard in-place). TODO: fix this properly by adding support for not in place input hadamard
        let (hidden, gate) = if let Some(gate_projection) = &self.gate_projection {
            let mut hidden_copy = encoder.allocate_scratch(hidden.size())?;
            encoder.encode_copy(&hidden, .., &mut hidden_copy, ..);
            let gate = gate_projection.encode(hidden, batch_dim.size(), encoder)?;
            (hidden_copy, Some(gate))
        } else {
            (hidden, None)
        };

        let mut attention_output = match state {
            Some(MaybeMut::Mut(state)) => {
                let qkv = self.qkv.project(hidden, batch_dim.size(), encoder)?;
                let queries = self.prepare_kv_and_queries(
                    &qkv,
                    state.keys.as_mut(),
                    state.values.as_mut(),
                    state.state_type.physical_prefix_length(),
                    self.num_q_heads,
                    precalculated_rope,
                    batch_dim.size(),
                    encoder,
                )?;
                self.run_core(&queries, batch_dim, state, encoder)?
            },
            Some(MaybeMut::Const(state)) => {
                // KV sharing: the packed projection produces queries only.
                let query = self.qkv.project(hidden, batch_dim.size(), encoder)?;
                let queries = self.prepare_queries(&query, precalculated_rope, batch_dim.size(), encoder)?;
                self.run_core(&queries, batch_dim, state, encoder)?
            },
            None => {
                let Some(num_kv_heads) = self.num_kv_heads else {
                    panic!("stateless attention doesn't support query-only projection");
                };
                assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

                let qkv = self.qkv.project(hidden, batch_dim.size(), encoder)?;
                let mut keys = encoder
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)?;
                let mut values = encoder
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)?;

                let queries = self.prepare_kv_and_queries(
                    &qkv,
                    &mut keys,
                    &mut values,
                    0,
                    self.num_q_heads,
                    precalculated_rope,
                    batch_dim.size(),
                    encoder,
                )?;

                // HACK: state_type should be Option.
                let state_type = if self.sliding_window_size.is_some() {
                    AttentionStateType::Ring {
                        offset: 0,
                        length: 0,
                        max_length: 0,
                    }
                } else {
                    AttentionStateType::Full {
                        length: 0,
                    }
                };

                self.kernel.encode(
                    AttentionArguments {
                        queries: &queries,
                        keys: &keys,
                        values: &values,
                        suffix_length: batch_dim.size(),
                        trie: None,
                        sinks: self.sinks.as_ref(),
                        state_type: &state_type,
                    },
                    encoder,
                )?
            },
        };

        if let Some(gate_kernel) = &self.gate_kernel {
            gate_kernel.encode(
                &gate.unwrap(),
                &mut attention_output,
                batch_dim.size() * (self.num_q_heads * self.head_dim),
                encoder,
            );
        }
        self.out_projection.encode(attention_output, batch_dim.size(), encoder)
    }

    pub fn append_projected_kv(
        &self,
        mut key_value: Allocation<B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: u32,
        state: &mut AttentionState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        if let Some(norm) = &self.qkv.norm {
            norm.encode_key_value(&mut key_value, batch_dim, encoder)?;
        }
        self.prepare_kv_and_queries(
            &key_value,
            state.keys.as_mut(),
            state.values.as_mut(),
            state.state_type.physical_prefix_length(),
            0,
            Some(precalculated_rope),
            batch_dim,
            encoder,
        )?;
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), encoder)?;
        Ok(())
    }

    fn run_core(
        &self,
        queries: &Allocation<B>,
        batch_dim: &BatchTopology,
        state: &AttentionState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let trie = if batch_dim.is_flat() {
            None
        } else {
            let mut trie = encoder.allocate_constant(batch_dim.size() as usize * size_of::<TrieNode>())?;
            trie.copyin(batch_dim.nodes());
            Some(trie)
        };

        self.kernel.encode(
            AttentionArguments {
                queries,
                keys: state.keys.as_ref(),
                values: state.values.as_ref(),
                suffix_length: batch_dim.size(),
                trie: trie.as_ref(),
                sinks: self.sinks.as_ref(),
                state_type: &state.state_type,
            },
            encoder,
        )
    }

    fn prepare_kv_and_queries<'keys, 'values>(
        &self,
        input: &Allocation<B>,
        keys: impl BufferArgMut<'keys, B>,
        values: impl BufferArgMut<'values, B>,
        kv_token_offset: u32,
        num_q_heads: u32,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut queries = if num_q_heads == 0 {
            encoder.allocate_scratch(self.data_type.size_in_bytes())?
        } else {
            encoder.allocate_scratch_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type)?
        };
        self.prepare.encode(
            input,
            &mut queries,
            Some(keys),
            Some(values),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
            num_q_heads,
            Some(self.num_kv_heads.expect("KV prepare requires KV heads")),
            self.head_dim,
            precalculated_rope.map(|precalculated_rope| precalculated_rope.dim),
            Some(kv_token_offset),
            batch_dim,
            encoder,
        );
        Ok(queries)
    }

    fn prepare_queries(
        &self,
        query: &Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut queries =
            encoder.allocate_scratch_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type)?;
        self.prepare.encode(
            query,
            &mut queries,
            None::<&mut Allocation<B>>,
            None::<&mut Allocation<B>>,
            precalculated_rope.map(|rope| &rope.cosines),
            precalculated_rope.map(|rope| &rope.sines),
            self.num_q_heads,
            None,
            self.head_dim,
            precalculated_rope.map(|rope| rope.dim),
            None,
            batch_dim,
            encoder,
        );
        Ok(queries)
    }
}
