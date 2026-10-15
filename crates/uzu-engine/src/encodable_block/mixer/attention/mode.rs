use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{
            AttentionArguments, AttentionInput, AttentionKernel, AttentionPrepareInt8Kernel, AttentionPrepareKernel,
            FullPrecisionInput, Int8Input,
        },
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::Linear,
        mixer::{
            MixerState,
            attention::{
                Attention, AttentionEncodeError, KVCacheView,
                kv_storage::{Int8KV, KVStorage},
                qkv_norm::QKVNorm,
                rope::PrecalculatedRoPE,
                state::AttentionState,
            },
        },
    },
    utils::maybe_mut::MaybeMut,
};

pub(super) struct LinearProjection<B: Backend> {
    pub(super) lin: Box<dyn Linear<B>>,
    pub(super) norm: QKVNorm<B>,
}

impl<B: Backend> LinearProjection<B> {
    fn project(
        &self,
        hidden: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let mut projected = self.lin.encode(hidden, batch_dim, command_buffer)?;
        self.norm.encode(&mut projected, batch_dim, command_buffer);
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
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let qkvg = self.projection.project(hidden, batch_dim.size(), command_buffer)?;
        let gate = self.has_gate.then(|| {
            qkvg.subrange(
                (self.projection_dim - self.num_q_heads * self.head_dim) as usize * self.data_type.size_in_bytes()..,
            )
        });

        let attention_output = match state {
            None => {
                let Some(num_kv_heads) = self.num_kv_heads else {
                    panic!("stateless attention doesn't support query-only projection");
                };
                assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

                let mut keys = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)
                    .map_err(AttentionEncodeError::Backend)?;
                let mut values = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)
                    .map_err(AttentionEncodeError::Backend)?;

                let queries = self
                    .prepare_kv_and_queries(
                        &qkvg,
                        &mut keys,
                        &mut values,
                        0,
                        self.num_q_heads,
                        precalculated_rope,
                        batch_dim.size(),
                        command_buffer,
                    )
                    .map_err(AttentionEncodeError::Backend)?;

                let cache = self.ring_capacity.map_or_else(|| KVCacheView::full(0), |_| KVCacheView::ring(0, 0));
                self.encode_bf16(&queries, &keys, &values, cache, gate, batch_dim, command_buffer)?
            },
            Some(MaybeMut::Mut(state)) => {
                let cache = state.view();
                match &mut state.kv {
                    KVStorage::BF16(kv, _) => {
                        let queries = self
                            .prepare_kv_and_queries(
                                &qkvg,
                                kv.keys.as_mut(),
                                kv.values.as_mut(),
                                cache.prefix_len(),
                                self.num_q_heads,
                                precalculated_rope,
                                batch_dim.size(),
                                command_buffer,
                            )
                            .map_err(AttentionEncodeError::Backend)?;
                        self.encode_bf16(
                            &queries,
                            kv.keys.as_ref(),
                            kv.values.as_ref(),
                            cache,
                            gate,
                            batch_dim,
                            command_buffer,
                        )?
                    },
                    KVStorage::Int8(kv, _) => {
                        let queries = self
                            .prepare_int8(&qkvg, Some(&mut *kv), cache, precalculated_rope, batch_dim, command_buffer)
                            .map_err(AttentionEncodeError::Backend)?;
                        self.encode_int8(queries, kv, cache, gate, batch_dim, command_buffer)?
                    },
                }
            },
            // KV sharing: QKVG contains queries and an optional gate only.
            Some(MaybeMut::Const(state)) => {
                let cache = state.view();
                match &state.kv {
                    KVStorage::BF16(kv, _) => {
                        let queries = self
                            .prepare_queries(&qkvg, precalculated_rope, batch_dim.size(), command_buffer)
                            .map_err(AttentionEncodeError::Backend)?;
                        self.encode_bf16(
                            &queries,
                            kv.keys.as_ref(),
                            kv.values.as_ref(),
                            cache,
                            gate,
                            batch_dim,
                            command_buffer,
                        )?
                    },
                    KVStorage::Int8(kv, _) => {
                        let queries = self
                            .prepare_int8(&qkvg, None, cache, precalculated_rope, batch_dim, command_buffer)
                            .map_err(AttentionEncodeError::Backend)?;
                        self.encode_int8(queries, kv, cache, gate, batch_dim, command_buffer)?
                    },
                }
            },
        };

        Ok(self.out_projection.encode(attention_output, batch_dim.size(), command_buffer)?)
    }

    pub fn append_projected_kv(
        &self,
        mut key_value: impl BufferMut<Backend = B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: u32,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.projection.norm.encode_key_value(key_value.reborrow(), batch_dim, command_buffer);
        let prefix_len = state.view().prefix_len();
        let KVStorage::BF16(kv, _) = &mut state.kv else {
            panic!("projected K/V rows go to a BF16 KV cache");
        };
        self.prepare_kv_and_queries(
            key_value.as_ref(),
            kv.keys.as_mut(),
            kv.values.as_mut(),
            prefix_len,
            0,
            Some(precalculated_rope),
            batch_dim,
            command_buffer,
        )?;
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), command_buffer)?;
        Ok(())
    }

    fn encode_bf16(
        &self,
        queries: impl BufferRef<Backend = B>,
        keys: impl BufferRef<Backend = B>,
        values: impl BufferRef<Backend = B>,
        cache: KVCacheView,
        gate: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let trie = Self::trie(batch_dim, command_buffer).map_err(AttentionEncodeError::Backend)?;
        self.kernel
            .encode(
                AttentionInput::FullPrecision(FullPrecisionInput {
                    queries,
                    keys,
                    values,
                }),
                AttentionArguments {
                    suffix_length: batch_dim.size(),
                    trie: trie.as_ref(),
                    sinks: self.sinks.as_ref(),
                    cache,
                    gate,
                },
                command_buffer,
            )
            .map_err(AttentionEncodeError::Kernel)
    }

    fn prepare_int8(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        kv: Option<&mut Int8KV<B>>,
        cache: KVCacheView,
        rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), B::Error> {
        let suffix_length = batch_dim.size();
        let query_rows = self.num_q_heads / self.num_groups * suffix_length;
        let mut query_codes =
            command_buffer.allocate_scratch_for_shape(&[self.num_groups, query_rows, self.head_dim], DataType::I8)?;
        let mut query_scales =
            command_buffer.allocate_scratch_for_shape(&[self.num_groups, query_rows], DataType::BF16)?;
        let (k_codes, k_scales, v_codes, v_scales) = match kv {
            Some(kv) => (
                Some(kv.k_codes.as_mut()),
                Some(kv.k_scales.as_mut()),
                Some(kv.v_codes.as_mut()),
                Some(kv.v_scales.as_mut()),
            ),
            None => (None, None, None, None),
        };
        self.prepare_int8.as_ref().expect("INT8 prepare kernel").encode(
            qkvg,
            None::<&mut B::ScratchBuffer>,
            Some(&mut query_codes),
            Some(&mut query_scales),
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            rope.map(|rope| &rope.cosines),
            rope.map(|rope| &rope.sines),
            self.num_q_heads,
            rope.map(|rope| rope.dim),
            cache.prefix_len(),
            self.projection_dim,
            suffix_length,
            query_rows,
            command_buffer,
        );
        Ok((query_codes, query_scales))
    }

    fn encode_int8(
        &self,
        (query_codes, query_scales): (B::ScratchBuffer, B::ScratchBuffer),
        kv: &Int8KV<B>,
        cache: KVCacheView,
        gate: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let trie = Self::trie(batch_dim, command_buffer).map_err(AttentionEncodeError::Backend)?;
        self.kernel
            .encode(
                AttentionInput::Int8(Int8Input {
                    query_codes: &query_codes,
                    query_scales: &query_scales,
                    k_codes: kv.k_codes.as_ref(),
                    k_scales: kv.k_scales.as_ref(),
                    v_codes: kv.v_codes.as_ref(),
                    v_scales: kv.v_scales.as_ref(),
                }),
                AttentionArguments {
                    sinks: self.sinks.as_ref(),
                    suffix_length: batch_dim.size(),
                    trie: trie.as_ref(),
                    cache,
                    gate,
                },
                command_buffer,
            )
            .map_err(AttentionEncodeError::Kernel)
    }

    fn trie(
        batch_dim: &BatchTopology,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<B::ConstantBuffer>, B::Error> {
        (!batch_dim.is_flat()).then(|| command_buffer.allocate_constant_from_slice(batch_dim.nodes())).transpose()
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
