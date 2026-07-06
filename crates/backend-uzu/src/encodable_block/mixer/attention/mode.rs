use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Buffer, Encoder, Kernels,
        gpu_types::trie::TrieNode,
        kernel::{AttentionPrepareKernel, SigmoidGateKernel},
    },
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::attention::{
            Attention,
            core::AttentionCoreEncodeArguments,
            projection::ProjectedQkv,
            rope::PrecalculatedRoPE,
            state::{AttentionState, AttentionStateType},
        },
    },
    utils::maybe_mut::MaybeMut,
};

pub(super) type AttentionPrepare<B> = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel;

pub(super) enum PrepareKernels<B: Backend> {
    Packed(AttentionPrepare<B>),
    #[allow(dead_code)]
    Split {
        query: AttentionPrepare<B>,
        key_value: AttentionPrepare<B>,
    },
}

impl<B: Backend> Attention<B> {
    pub(super) fn encode_attend(
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

        let projected = self.projection.encode_attend(hidden, batch_dim.size(), encoder)?;
        let mut attention_output = match (projected, state) {
            (ProjectedQkv::Fused(qkv), Some(MaybeMut::Mut(mut state))) => {
                self.encode_with_owned_cache(qkv, precalculated_rope, batch_dim, &mut state, encoder)?
            },
            (ProjectedQkv::Query(query), Some(MaybeMut::Const(state))) => {
                self.encode_with_borrowed_cache(query, precalculated_rope, batch_dim, state, encoder)?
            },
            (
                ProjectedQkv::Split {
                    query,
                    key_value,
                },
                Some(MaybeMut::Mut(mut state)),
            ) => self.encode_with_ephemeral_suffix(
                query,
                key_value,
                precalculated_rope.expect("split attention requires RoPE"),
                batch_dim,
                &mut state,
                encoder,
            )?,
            (ProjectedQkv::Fused(qkv), None) => self.encode_stateless(qkv, precalculated_rope, batch_dim, encoder)?,
            _ => panic!("attention projection/state combination is invalid"),
        };

        if let Some(gate_kernel) = &self.gate_kernel {
            gate_kernel.encode(
                &gate.unwrap(),
                &mut attention_output,
                (batch_dim.size() * self.num_q_heads * self.head_dim) as u32,
                encoder,
            );
        }
        self.out_projection.encode(attention_output, batch_dim.size(), encoder)
    }

    // Split append is exercised here and becomes reachable when the split-attention constructor lands.
    #[allow(dead_code)]
    pub(super) fn encode_append(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: usize,
        state: &mut AttentionState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let key_value = self.projection.encode_key_value(hidden, batch_dim, encoder)?;
        self.prepare_split_key_value(
            key_value,
            precalculated_rope,
            batch_dim,
            state.state_type.physical_prefix_length(),
            state.keys.as_mut(),
            state.values.as_mut(),
            encoder,
        )?;
        state.append_full(batch_dim);
        Ok(())
    }

    pub(super) fn encode_with_ephemeral_suffix(
        &self,
        query: Allocation<B>,
        key_value: Allocation<B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: &BatchTopology,
        state: &mut AttentionState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let queries = self.prepare_split_query(query, precalculated_rope, batch_dim.size(), encoder)?;
        self.prepare_split_key_value(
            key_value,
            precalculated_rope,
            batch_dim.size(),
            state.state_type.physical_prefix_length(),
            state.keys.as_mut(),
            state.values.as_mut(),
            encoder,
        )?;
        self.encode_prepared_queries(&queries, batch_dim, state, encoder)
    }

    fn encode_with_owned_cache(
        &self,
        qkv: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: &mut AttentionState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut queries = self.allocate_queries(batch_dim.size(), encoder)?;
        self.packed_prepare().encode(
            &qkv,
            &mut queries,
            Some(state.keys.as_mut()),
            Some(state.values.as_mut()),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
            self.num_q_heads as u32,
            self.num_kv_heads.map(|num_kv_heads| num_kv_heads as u32),
            self.head_dim as u32,
            precalculated_rope.map(|precalculated_rope| precalculated_rope.dim as u32),
            Some(state.state_type.physical_prefix_length() as u32),
            batch_dim.size() as u32,
            encoder,
        );
        self.encode_prepared_queries(&queries, batch_dim, state, encoder)
    }

    fn encode_with_borrowed_cache(
        &self,
        query: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: &AttentionState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut queries = self.allocate_queries(batch_dim.size(), encoder)?;
        self.packed_prepare().encode(
            &query,
            &mut queries,
            None::<&mut Allocation<B>>,
            None::<&mut Allocation<B>>,
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
            self.num_q_heads as u32,
            None,
            self.head_dim as u32,
            precalculated_rope.map(|precalculated_rope| precalculated_rope.dim as u32),
            None,
            batch_dim.size() as u32,
            encoder,
        );
        self.encode_prepared_queries(&queries, batch_dim, state, encoder)
    }

    fn encode_stateless(
        &self,
        qkv: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let Some(num_kv_heads) = self.num_kv_heads else {
            panic!("stateless attention doesn't support query-only projection");
        };

        assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

        let mut queries = self.allocate_queries(batch_dim.size(), encoder)?;
        let mut keys = encoder
            .allocate_scratch(size_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type))?;
        let mut values = encoder
            .allocate_scratch(size_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type))?;

        self.packed_prepare().encode(
            &qkv,
            &mut queries,
            Some(&mut keys),
            Some(&mut values),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
            self.num_q_heads as u32,
            Some(num_kv_heads as u32),
            self.head_dim as u32,
            precalculated_rope.map(|precalculated_rope| precalculated_rope.dim as u32),
            Some(0),
            batch_dim.size() as u32,
            encoder,
        );

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

        self.flat_core.encode(
            AttentionCoreEncodeArguments {
                queries: &queries,
                keys: &keys,
                values: &values,
                suffix_length: batch_dim.size(),
                trie: None,
                sinks: self.sinks.as_ref(),
                state_type: &state_type,
            },
            encoder,
        )
    }

    fn encode_prepared_queries(
        &self,
        queries: &Allocation<B>,
        batch_dim: &BatchTopology,
        state: &AttentionState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let (core, trie) = if batch_dim.is_flat() {
            (&self.flat_core, None)
        } else {
            let mut trie = encoder.allocate_constant(batch_dim.size() * size_of::<TrieNode>())?;
            trie.copyin(batch_dim.nodes());
            (&self.trie_core, Some(trie))
        };

        core.encode(
            AttentionCoreEncodeArguments {
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

    fn allocate_queries(
        &self,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.allocate_scratch(size_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type))
    }

    fn packed_prepare(&self) -> &AttentionPrepare<B> {
        let PrepareKernels::Packed(prepare) = &self.prepare else {
            panic!("packed QKV prepare kernel is not available");
        };
        prepare
    }

    fn prepare_split_query(
        &self,
        query: Allocation<B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let PrepareKernels::Split {
            query: prepare,
            ..
        } = &self.prepare
        else {
            panic!("split query prepare kernel is not available");
        };
        let mut queries = self.allocate_queries(batch_dim, encoder)?;
        prepare.encode(
            &query,
            &mut queries,
            None::<&mut Allocation<B>>,
            None::<&mut Allocation<B>>,
            Some(&precalculated_rope.cosines),
            Some(&precalculated_rope.sines),
            self.num_q_heads as u32,
            None,
            self.head_dim as u32,
            Some(precalculated_rope.dim as u32),
            None,
            batch_dim as u32,
            encoder,
        );
        Ok(queries)
    }

    fn prepare_split_key_value(
        &self,
        key_value: Allocation<B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: usize,
        key_value_offset: usize,
        keys: &mut dyn Buffer<Backend = B>,
        values: &mut dyn Buffer<Backend = B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let PrepareKernels::Split {
            key_value: prepare,
            ..
        } = &self.prepare
        else {
            panic!("split key-value prepare kernel is not available");
        };
        let num_kv_heads = self.num_kv_heads.expect("split key-value prepare requires KV heads");
        let mut unused_queries = encoder.allocate_scratch(self.data_type.size_in_bytes())?;
        prepare.encode(
            &key_value,
            &mut unused_queries,
            Some(keys),
            Some(values),
            Some(&precalculated_rope.cosines),
            Some(&precalculated_rope.sines),
            0,
            Some(num_kv_heads as u32),
            self.head_dim as u32,
            Some(precalculated_rope.dim as u32),
            Some(key_value_offset as u32),
            batch_dim as u32,
            encoder,
        );
        Ok(())
    }
}
