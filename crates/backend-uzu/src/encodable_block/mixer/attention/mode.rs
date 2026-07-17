use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, BufferArgMut, Encoder, Kernels,
        gpu_types::trie::TrieNode,
        kernel::{AttentionLastQueryKernel, AttentionPrepareKernel, SigmoidGateKernel},
    },
    encodable_block::{
        batch_topology::BatchTopology,
        linear::Linear,
        mixer::attention::{
            Attention,
            core::AttentionCoreEncodeArguments,
            qkv_norm::QKVNorm,
            rope::PrecalculatedRoPE,
            state::{AttentionState, AttentionStateType},
        },
    },
    utils::maybe_mut::MaybeMut,
};

type PrepareKernel<B> = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel;

pub(super) struct LinearProjection<B: Backend> {
    pub(super) lin: Box<dyn Linear<B>>,
    pub(super) norm: Option<QKVNorm<B>>,
}

impl<B: Backend> LinearProjection<B> {
    fn project(
        &self,
        hidden: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut projected = self.lin.encode(hidden, batch_dim, encoder)?;
        if let Some(norm) = &self.norm {
            norm.encode(&mut projected, batch_dim, encoder)?;
        }
        Ok(projected)
    }
}

pub(super) enum QkvProjection<B: Backend> {
    /// Fused QKV — or Q-only under KV sharing (`num_kv_heads == None`).
    Packed {
        qkv: LinearProjection<B>,
        prepare: PrepareKernel<B>,
    },
    /// Separate Q and KV projections.
    #[allow(dead_code)] // Retained for model families with separate Q and KV weights.
    Split {
        q: LinearProjection<B>,
        kv: LinearProjection<B>,
        q_prepare: PrepareKernel<B>,
        kv_prepare: PrepareKernel<B>,
    },
}

impl<B: Backend> Attention<B> {
    pub(super) fn attend_packed_last_queries(
        &self,
        hidden: Allocation<B>,
        lengths: &Allocation<B>,
        rows: usize,
        sequence_length: usize,
        scale: f32,
        kernel: &<B::Kernels as Kernels>::AttentionLastQueryKernel,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let QkvProjection::Packed {
            qkv,
            ..
        } = &self.projection
        else {
            panic!("packed last-query attention requires packed QKV");
        };
        assert_eq!(self.num_kv_heads, Some(self.num_q_heads));
        let qkv = qkv.project(hidden, rows * sequence_length, encoder)?;
        let mut attention_output =
            encoder.allocate_scratch(size_for_shape(&[rows, self.num_q_heads, self.head_dim], self.data_type))?;
        kernel.encode(
            &qkv,
            lengths,
            &mut attention_output,
            rows as u32,
            sequence_length as u32,
            self.num_q_heads as u32,
            self.head_dim as u32,
            scale,
            encoder,
        );
        self.out_projection.encode(attention_output, rows, encoder)
    }

    pub(super) fn attend(
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

        let mut attention_output = match (&self.projection, state) {
            (
                QkvProjection::Packed {
                    qkv,
                    prepare,
                },
                Some(MaybeMut::Mut(state)),
            ) => {
                let qkv = qkv.project(hidden, batch_dim.size(), encoder)?;
                let queries = self.prepare_kv_and_queries(
                    prepare,
                    &qkv,
                    state.keys.as_mut(),
                    state.values.as_mut(),
                    state.state_type.physical_prefix_length(),
                    self.num_q_heads as u32,
                    precalculated_rope,
                    batch_dim.size(),
                    encoder,
                )?;
                self.run_core(&queries, batch_dim, state, encoder)?
            },
            (
                QkvProjection::Packed {
                    qkv,
                    prepare,
                },
                Some(MaybeMut::Const(state)),
            ) => {
                // KV sharing: the packed projection produces queries only.
                let query = qkv.project(hidden, batch_dim.size(), encoder)?;
                let queries = self.prepare_queries(prepare, &query, precalculated_rope, batch_dim.size(), encoder)?;
                self.run_core(&queries, batch_dim, state, encoder)?
            },
            (
                QkvProjection::Packed {
                    qkv,
                    prepare,
                },
                None,
            ) => {
                let Some(num_kv_heads) = self.num_kv_heads else {
                    panic!("stateless attention doesn't support query-only projection");
                };
                assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

                let qkv = qkv.project(hidden, batch_dim.size(), encoder)?;
                let mut keys = encoder.allocate_scratch(size_for_shape(
                    &[batch_dim.size(), num_kv_heads, self.head_dim],
                    self.data_type,
                ))?;
                let mut values = encoder.allocate_scratch(size_for_shape(
                    &[batch_dim.size(), num_kv_heads, self.head_dim],
                    self.data_type,
                ))?;

                let queries = self.prepare_kv_and_queries(
                    prepare,
                    &qkv,
                    &mut keys,
                    &mut values,
                    0,
                    self.num_q_heads as u32,
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
                )?
            },
            (
                QkvProjection::Split {
                    q,
                    kv,
                    q_prepare,
                    kv_prepare,
                },
                Some(MaybeMut::Mut(state)),
            ) => {
                // Linear::encode may consume/mutate its input; split Q/KV attention needs the same hidden for both projections.
                let mut hidden_for_key_value = encoder.allocate_scratch(hidden.size())?;
                encoder.encode_copy(&hidden, .., &mut hidden_for_key_value, ..);
                let query = q.project(hidden, batch_dim.size(), encoder)?;
                let key_value = kv.project(hidden_for_key_value, batch_dim.size(), encoder)?;
                let precalculated_rope = precalculated_rope.expect("split attention requires RoPE");
                let queries =
                    self.prepare_queries(q_prepare, &query, Some(precalculated_rope), batch_dim.size(), encoder)?;
                self.prepare_kv_and_queries(
                    kv_prepare,
                    &key_value,
                    state.keys.as_mut(),
                    state.values.as_mut(),
                    state.state_type.physical_prefix_length(),
                    0,
                    Some(precalculated_rope),
                    batch_dim.size(),
                    encoder,
                )?;
                self.run_core(&queries, batch_dim, state, encoder)?
            },
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

    pub(super) fn append_kv(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: usize,
        state: &mut AttentionState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        match &self.projection {
            QkvProjection::Split {
                kv,
                kv_prepare,
                ..
            } => {
                let key_value = kv.project(hidden, batch_dim, encoder)?;
                self.prepare_kv_and_queries(
                    kv_prepare,
                    &key_value,
                    state.keys.as_mut(),
                    state.values.as_mut(),
                    state.state_type.physical_prefix_length(),
                    0,
                    precalculated_rope,
                    batch_dim,
                    encoder,
                )?;
            },
            QkvProjection::Packed {
                qkv,
                prepare,
            } => {
                let projected = qkv.project(hidden, batch_dim, encoder)?;
                let num_kv_heads = self.num_kv_heads.expect("packed KV append requires separate KV heads");
                let q_elements = self.num_q_heads * self.head_dim;
                let kv_elements = num_kv_heads * self.head_dim;
                let element_size = self.data_type.size_in_bytes();
                let mut key_value =
                    encoder.allocate_scratch(size_for_shape(&[batch_dim, 2 * kv_elements], self.data_type))?;
                let projected_row_elements = q_elements + 2 * kv_elements;
                for row in 0..batch_dim {
                    let source_start = (row * projected_row_elements + q_elements) * element_size;
                    let source_end = source_start + 2 * kv_elements * element_size;
                    let destination_start = row * 2 * kv_elements * element_size;
                    let destination_end = destination_start + 2 * kv_elements * element_size;
                    encoder.encode_copy(
                        &projected,
                        source_start..source_end,
                        &mut key_value,
                        destination_start..destination_end,
                    );
                }
                self.prepare_kv_and_queries(
                    prepare,
                    &key_value,
                    state.keys.as_mut(),
                    state.values.as_mut(),
                    state.state_type.physical_prefix_length(),
                    0,
                    precalculated_rope,
                    batch_dim,
                    encoder,
                )?;
            },
        }
        state.append_full(batch_dim);
        Ok(())
    }

    fn run_core(
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

    /// With `num_q_heads == 0`, only keys/values are scattered into the cache (KV append).
    fn prepare_kv_and_queries<'keys, 'values>(
        &self,
        prepare: &PrepareKernel<B>,
        input: &Allocation<B>,
        keys: impl BufferArgMut<'keys, B>,
        values: impl BufferArgMut<'values, B>,
        kv_token_offset: usize,
        num_q_heads: u32,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut queries = if num_q_heads == 0 {
            encoder.allocate_scratch(self.data_type.size_in_bytes())?
        } else {
            self.allocate_queries(batch_dim, encoder)?
        };
        prepare.encode(
            input,
            &mut queries,
            Some(keys),
            Some(values),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
            num_q_heads,
            Some(self.num_kv_heads.expect("KV prepare requires KV heads") as u32),
            self.head_dim as u32,
            precalculated_rope.map(|precalculated_rope| precalculated_rope.dim as u32),
            Some(kv_token_offset as u32),
            batch_dim as u32,
            encoder,
        );
        Ok(queries)
    }

    fn prepare_queries(
        &self,
        prepare: &PrepareKernel<B>,
        query: &Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut queries = self.allocate_queries(batch_dim, encoder)?;
        prepare.encode(
            query,
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
            batch_dim as u32,
            encoder,
        );
        Ok(queries)
    }

    fn allocate_queries(
        &self,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.allocate_scratch(size_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type))
    }
}
