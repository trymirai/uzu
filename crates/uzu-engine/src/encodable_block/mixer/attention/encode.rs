use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionPrepareKernel},
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{
            MixerState,
            attention::{
                Attention, AttentionEncodeError, Int8KV, KVCacheView, rope::PrecalculatedRoPE, state::AttentionState,
            },
        },
    },
    utils::maybe_mut::MaybeMut,
};

impl<B: Backend> Attention<B> {
    pub(super) fn attend(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let qkvg = self.projection.encode(hidden, batch_dim.size(), command_buffer)?;
        let gate = self.has_gate.then(|| {
            qkvg.subrange(
                (self.projection_dim - self.num_q_heads * self.head_dim) as usize * self.data_type.size_in_bytes()..,
            )
        });

        let attention_output = {
            let mut scratch_kv;
            let (mut kv, cache) = match state {
                Some(MaybeMut::Mut(state)) => {
                    let cache = state.view();
                    (MaybeMut::Mut(&mut state.kv), cache)
                },
                Some(MaybeMut::Const(state)) => (MaybeMut::Const(&state.kv), state.view()),
                None => {
                    assert!(self.num_kv_heads.is_some(), "stateless query-only attention");
                    assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");
                    scratch_kv = Int8KV::allocate_scratch(self.kernel.kv_layout(), batch_dim.size(), command_buffer)
                        .map_err(AttentionEncodeError::Backend)?;
                    let cache = self.ring_capacity.map_or_else(|| KVCacheView::full(0), |_| KVCacheView::ring(0, 0));
                    (MaybeMut::Mut(&mut scratch_kv), cache)
                },
            };
            let (queries, query_scales) = self
                .prepare(
                    &qkvg,
                    &mut kv,
                    cache.prefix_len(),
                    self.num_q_heads,
                    precalculated_rope,
                    batch_dim.size(),
                    command_buffer,
                )
                .map_err(AttentionEncodeError::Backend)?;
            let input = AttentionInput {
                query_codes: &queries,
                query_scales: &query_scales,
                k_codes: kv.k_codes.as_ref(),
                k_scales: kv.k_scales.as_ref(),
                v_codes: kv.v_codes.as_ref(),
                v_scales: kv.v_scales.as_ref(),
            };
            let trie = (!batch_dim.is_flat())
                .then(|| command_buffer.allocate_constant_from_slice(batch_dim.nodes()))
                .transpose()
                .map_err(AttentionEncodeError::Backend)?;
            self.kernel
                .encode(
                    input,
                    AttentionArguments {
                        suffix_length: batch_dim.size(),
                        trie: trie.as_ref(),
                        sinks: self.sinks.as_ref(),
                        cache,
                        gate,
                    },
                    command_buffer,
                )
                .map_err(AttentionEncodeError::Kernel)?
        };

        Ok(self.out_projection.encode(attention_output, batch_dim.size(), command_buffer)?)
    }

    pub fn append_projected_kv(
        &self,
        key_value: impl BufferRef<Backend = B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: u32,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let prefix_len = state.view().prefix_len();
        self.prepare(
            key_value,
            &mut MaybeMut::Mut(&mut state.kv),
            prefix_len,
            0,
            Some(precalculated_rope),
            batch_dim,
            command_buffer,
        )?;
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), command_buffer)?;
        Ok(())
    }

    fn prepare(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        kv: &mut MaybeMut<'_, Int8KV<B>>,
        kv_token_offset: u32,
        num_q_heads: u32,
        rope: Option<&PrecalculatedRoPE<B>>,
        suffix_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), B::Error> {
        assert_eq!(kv.layout, self.kernel.kv_layout(), "shared KV layout mismatch");
        let (k_codes, k_scales, v_codes, v_scales) = match kv {
            MaybeMut::Mut(kv) => (
                Some(kv.k_codes.as_mut()),
                Some(kv.k_scales.as_mut()),
                Some(kv.v_codes.as_mut()),
                Some(kv.v_scales.as_mut()),
            ),
            MaybeMut::Const(_) => (None, None, None, None),
        };
        let input_row_stride = if num_q_heads == 0 {
            2 * self.num_groups * self.head_dim
        } else {
            self.projection_dim
        };
        let query_count = num_q_heads as usize * suffix_length as usize;
        // K/V-only calls still need bound query buffers.
        let mut query_codes = command_buffer.allocate_scratch((query_count * self.head_dim as usize).max(1))?;
        let mut query_scales = command_buffer.allocate_scratch(query_count.max(1) * DataType::BF16.size_in_bytes())?;
        self.prepare.encode(
            qkvg,
            &mut query_codes,
            &mut query_scales,
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            rope.map(|rope| &rope.cosines),
            rope.map(|rope| &rope.sines),
            self.qkv_norm.scales[0].as_ref(),
            self.qkv_norm.scales[1].as_ref(),
            self.qkv_norm.scales[2].as_ref(),
            self.qkv_norm.params,
            num_q_heads,
            rope.map(|rope| rope.dim),
            kv_token_offset,
            input_row_stride,
            suffix_length,
            num_q_heads / self.num_groups * suffix_length,
            command_buffer,
        );
        Ok((query_codes, query_scales))
    }
}
