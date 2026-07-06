use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{Allocation, Backend, Buffer, Encoder, Kernels, kernel::AttentionPrepareKernel},
    config::dflash::DFlashAttentionConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        mixer::{
            MixerState,
            attention::{
                core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments, AttentionCores},
                qkv_norm::{QKVNorm, QKVNormError},
                rope::PrecalculatedRoPE,
                state::AttentionState,
            },
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub(crate) enum DFlashAttentionError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("QKVNorm error: {0}")]
    QKVNorm(#[from] QKVNormError<B>),
}

pub(crate) struct DFlashAttention<B: Backend> {
    hidden_dim: usize,
    data_type: DataType,
    query_projection: Box<dyn Linear<B>>,
    key_value_projection: Box<dyn Linear<B>>,
    query_norm: QKVNorm<B>,
    key_norm: QKVNorm<B>,
    core: DFlashAttentionCore<B>,
    out_projection: Box<dyn Linear<B>>,
}

struct DFlashAttentionCore<B: Backend> {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    max_rope_length: usize,
    data_type: DataType,
    query_prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    key_value_prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    attention_core: AttentionCores<B>,
}

impl<B: Backend> DFlashAttention<B> {
    pub(crate) fn new(
        hidden_dim: usize,
        data_type: DataType,
        config: &DFlashAttentionConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, DFlashAttentionError<B>> {
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;

        let query_projection = <dyn Linear<B>>::new(
            hidden_dim,
            [q_dim],
            config.has_attention_biases,
            context,
            data_type,
            &parameter_tree.subtree("query_projection")?,
        )?;
        let key_value_projection = <dyn Linear<B>>::new(
            hidden_dim,
            [kv_dim * 2],
            config.has_attention_biases,
            context,
            data_type,
            &parameter_tree.subtree("key_value_projection")?,
        )?;

        let query_norm = QKVNorm::new(
            context,
            data_type,
            Some(config.query_norm_config.clone()),
            None,
            None,
            parameter_tree,
            config.num_heads,
            0,
            config.head_dim,
        )?;
        let key_norm = QKVNorm::new(
            context,
            data_type,
            None,
            Some(config.key_norm_config.clone()),
            None,
            parameter_tree,
            0,
            config.num_key_value_heads,
            config.head_dim,
        )?;

        let core = DFlashAttentionCore::new(
            config.head_dim,
            config.num_heads,
            config.num_key_value_heads,
            *config.rope_config.max_sequence_length(),
            config.sliding_window_size,
            config.scale,
            data_type,
            context,
        )
        .map_err(DFlashAttentionError::Backend)?;

        let out_projection = <dyn Linear<B>>::new(
            q_dim,
            [hidden_dim],
            config.has_output_biases,
            context,
            data_type,
            &parameter_tree.subtree("out_projection")?,
        )?;

        Ok(Self {
            hidden_dim,
            data_type,
            query_projection,
            key_value_projection,
            query_norm,
            key_norm,
            core,
            out_projection,
        })
    }

    pub(crate) fn create_empty_state(
        &self,
        max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<AttentionState<B>, B::Error> {
        self.core.create_empty_state(max_context_length, context)
    }

    pub(crate) fn encode_context_append(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        state: &mut AttentionState<B>,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let batch_dim = self.batch_dim(&hidden);
        let mut key_value = self.key_value_projection.encode(hidden, batch_dim, encoder)?;
        self.key_norm.encode(&mut key_value, batch_dim, encoder)?;
        self.core.encode_context_append(&mut key_value, batch_dim, precalculated_rope, state, context, encoder)
    }

    pub(crate) fn encode_draft(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        state: &mut AttentionState<B>,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let batch_dim = self.batch_dim(&hidden);
        let mut hidden_for_kv = encoder.allocate_scratch(hidden.size())?;
        encoder.encode_copy(&hidden, .., &mut hidden_for_kv, ..);

        let mut query = self.query_projection.encode(hidden, batch_dim, encoder)?;
        let mut key_value = self.key_value_projection.encode(hidden_for_kv, batch_dim, encoder)?;
        self.query_norm.encode(&mut query, batch_dim, encoder)?;
        self.key_norm.encode(&mut key_value, batch_dim, encoder)?;
        let attention_output = self.core.encode_draft(
            &mut query,
            &mut key_value,
            batch_dim,
            precalculated_rope,
            state,
            context,
            encoder,
        )?;
        self.out_projection.encode(attention_output, batch_dim, encoder)
    }

    fn batch_dim(
        &self,
        hidden: &Allocation<B>,
    ) -> usize {
        let row_bytes = size_for_shape(&[self.hidden_dim], self.data_type);
        assert_eq!(hidden.size() % row_bytes, 0, "DFlash attention input has partial row");
        hidden.size() / row_bytes
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use proc_macros::uzu_test;

    use super::DFlashAttentionCore;
    use crate::{
        backends::{
            common::{Backend, Buffer, Context, Encoder},
            cpu::Cpu,
        },
        data_type::DataType,
        encodable_block::mixer::attention::{rope::PrecalculatedRoPE, state::AttentionStateType},
        tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, submit_encoder},
    };

    const HEAD_DIM: usize = 64;

    fn core(
        context: &<Cpu as Backend>::Context,
        sliding_window_size: Option<usize>,
    ) -> DFlashAttentionCore<Cpu> {
        DFlashAttentionCore::new(HEAD_DIM, 2, 1, 16, sliding_window_size, 1.0, DataType::BF16, context).unwrap()
    }

    fn identity_rope(
        context: &<Cpu as Backend>::Context,
        tokens: usize,
    ) -> PrecalculatedRoPE<Cpu> {
        PrecalculatedRoPE {
            cosines: alloc_allocation_with_data(context, &vec![1.0_f32; tokens * HEAD_DIM]),
            sines: alloc_allocation_with_data(context, &vec![0.0_f32; tokens * HEAD_DIM]),
            dim: HEAD_DIM,
        }
    }

    fn bf16s(values: Vec<f32>) -> Vec<bf16> {
        values.iter().copied().map(bf16::from_f32).collect()
    }

    fn row(values: [f32; 4]) -> [f32; HEAD_DIM] {
        let mut row = [0.0; HEAD_DIM];
        row[..4].copy_from_slice(&values);
        row
    }

    fn projected_kv(rows: &[([f32; 4], [f32; 4])]) -> Vec<f32> {
        rows.iter().flat_map(|(key, value)| row(*key).into_iter().chain(row(*value))).collect()
    }

    fn projected_queries(rows: &[[f32; 4]]) -> Vec<f32> {
        rows.iter().flat_map(|values| row(*values)).collect()
    }

    fn first4_by_row(values: &[f32]) -> Vec<[f32; 4]> {
        values.chunks_exact(HEAD_DIM).map(|row| row[..4].try_into().unwrap()).collect()
    }

    fn read_cpu_buffer(
        buffer: &dyn Buffer<Backend = Cpu>,
        elements: usize,
    ) -> Vec<f32> {
        let dense = buffer.downcast();
        let bytes = unsafe { &*dense.get() };
        bytemuck::cast_slice::<u8, bf16>(&bytes[..elements * size_of::<bf16>()])
            .iter()
            .map(|value| value.to_f32())
            .collect()
    }

    #[uzu_test]
    fn context_append_writes_key_values_and_advances_full_state() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let context = context.as_ref();
        let core = core(context, None);
        let rope = identity_rope(context, 3);
        let mut state = core.create_empty_state(Some(8), context).unwrap();
        let mut key_value = alloc_allocation_with_data(
            context,
            &bf16s(projected_kv(&[
                ([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]),
                ([5.0, 6.0, 7.0, 8.0], [50.0, 60.0, 70.0, 80.0]),
                ([9.0, 10.0, 11.0, 12.0], [90.0, 100.0, 110.0, 120.0]),
            ])),
        );

        let mut encoder = Encoder::new(context).unwrap();
        core.encode_context_append(&mut key_value, 3, &rope, &mut state, context, &mut encoder).unwrap();

        assert_eq!(state.cur_context_length, 3);
        assert!(matches!(
            state.state_type,
            AttentionStateType::Full {
                length: 3
            }
        ));

        submit_encoder(encoder);
        assert_eq!(
            first4_by_row(&read_cpu_buffer(state.keys.as_ref(), 3 * HEAD_DIM)),
            vec![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        );
        assert_eq!(
            first4_by_row(&read_cpu_buffer(state.values.as_ref(), 3 * HEAD_DIM)),
            vec![[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0], [90.0, 100.0, 110.0, 120.0]]
        );
    }

    #[uzu_test]
    fn draft_encode_is_non_causal_and_keeps_committed_length() {
        let (actual, expected) = draft_case(None);
        assert_close(&actual, &expected);
    }

    #[uzu_test]
    fn draft_encode_applies_symmetric_sliding_window() {
        let (actual, expected) = draft_case(Some(2));
        assert_close(&actual, &expected);
    }

    fn draft_case(sliding_window_size: Option<usize>) -> (Vec<f32>, Vec<f32>) {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let context = context.as_ref();
        let core = core(context, sliding_window_size);
        let rope = identity_rope(context, 2);
        let mut state = core.create_empty_state(Some(8), context).unwrap();
        let mut prefix = alloc_allocation_with_data(
            context,
            &bf16s(projected_kv(&[
                ([1.0, 0.0, 0.0, 0.0], [1.0, 10.0, 0.0, 0.0]),
                ([0.0, 1.0, 0.0, 0.0], [2.0, 20.0, 0.0, 0.0]),
            ])),
        );
        let mut encoder = Encoder::new(context).unwrap();
        core.encode_context_append(&mut prefix, 2, &rope, &mut state, context, &mut encoder).unwrap();
        submit_encoder(encoder);

        let mut query = alloc_allocation_with_data(
            context,
            &bf16s(projected_queries(&[
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ])),
        );
        let mut suffix = alloc_allocation_with_data(
            context,
            &bf16s(projected_kv(&[
                ([0.0, 0.0, 1.0, 0.0], [3.0, 30.0, 0.0, 0.0]),
                ([0.0, 0.0, 0.0, 1.0], [4.0, 40.0, 0.0, 0.0]),
            ])),
        );

        let mut encoder = Encoder::new(context).unwrap();
        let output = core.encode_draft(&mut query, &mut suffix, 2, &rope, &mut state, context, &mut encoder).unwrap();
        let mut output_copy = alloc_allocation::<Cpu, bf16>(context, output.size() / size_of::<bf16>());
        encoder.encode_copy(&output, .., &mut output_copy, ..);
        drop(output);
        assert_eq!(state.cur_context_length, 2);
        assert!(matches!(
            state.state_type,
            AttentionStateType::Full {
                length: 2
            }
        ));
        submit_encoder(encoder);

        let actual =
            allocation_to_vec::<Cpu, bf16>(&output_copy).iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let expected = reference_attention(
            &projected_queries(&[
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            &projected_queries(&[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            &projected_queries(&[
                [1.0, 10.0, 0.0, 0.0],
                [2.0, 20.0, 0.0, 0.0],
                [3.0, 30.0, 0.0, 0.0],
                [4.0, 40.0, 0.0, 0.0],
            ]),
            2,
            sliding_window_size,
        );
        (actual, expected)
    }

    fn reference_attention(
        queries_by_head: &[f32],
        keys: &[f32],
        values: &[f32],
        suffix_length: usize,
        sliding_window_size: Option<usize>,
    ) -> Vec<f32> {
        let key_count = keys.len() / HEAD_DIM;
        let prefix_length = key_count - suffix_length;
        let mut output = Vec::new();
        for query_index in 0..suffix_length {
            for head_index in 0..2 {
                let query_offset = (query_index * 2 + head_index) * HEAD_DIM;
                let query = &queries_by_head[query_offset..query_offset + HEAD_DIM];
                let query_position = prefix_length + query_index;
                let mut scores = Vec::new();
                for key_position in 0..key_count {
                    let in_window =
                        sliding_window_size.is_none_or(|window| key_position.abs_diff(query_position) <= window / 2);
                    if in_window {
                        let key_offset = key_position * HEAD_DIM;
                        let key = &keys[key_offset..key_offset + HEAD_DIM];
                        scores.push((key_position, query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>()));
                    }
                }
                let max_score = scores.iter().map(|(_, score)| *score).fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores.iter().map(|(_, score)| (*score - max_score).exp()).sum::<f32>();
                let mut row = [0.0; HEAD_DIM];
                for (key_position, score) in scores {
                    let weight = (score - max_score).exp() / denominator;
                    let value_offset = key_position * HEAD_DIM;
                    let value = &values[value_offset..value_offset + HEAD_DIM];
                    for (out, value) in row.iter_mut().zip(value) {
                        *out += weight * value;
                    }
                }
                output.extend(row);
            }
        }
        output
    }

    fn assert_close(
        actual: &[f32],
        expected: &[f32],
    ) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() < 0.08, "mismatch at {index}: actual={actual}, expected={expected}");
        }
    }
}

impl<B: Backend> DFlashAttentionCore<B> {
    fn new(
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        max_rope_length: usize,
        sliding_window_size: Option<usize>,
        scale: f32,
        data_type: DataType,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let query_prepare =
            <B::Kernels as Kernels>::AttentionPrepareKernel::new(context, data_type, DataType::F32, false, true)?;
        let key_value_prepare =
            <B::Kernels as Kernels>::AttentionPrepareKernel::new(context, data_type, DataType::F32, true, true)?;
        let attention_core = AttentionCores::new(
            AttentionCoreNewArguments {
                head_dim,
                num_groups: num_kv_heads,
                num_q_heads,
                has_sinks: false,
                is_kv_cache_ring: false,
                is_causal: false,
                is_trie: false,
                sliding_window_size,
                scale: Some(scale),
                data_type,
            },
            context,
        )?;

        Ok(Self {
            head_dim,
            num_q_heads,
            num_kv_heads,
            max_rope_length,
            data_type,
            query_prepare,
            key_value_prepare,
            attention_core,
        })
    }

    fn create_empty_state(
        &self,
        max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<AttentionState<B>, B::Error> {
        AttentionState::create_empty_full(
            max_context_length,
            self.max_rope_length,
            self.data_type,
            self.num_kv_heads,
            self.head_dim,
            context,
        )
    }

    fn encode_context_append(
        &self,
        key_value: &mut Allocation<B>,
        batch_dim: usize,
        precalculated_rope: &PrecalculatedRoPE<B>,
        state: &mut AttentionState<B>,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        state.prepare(state.cur_context_length, batch_dim, context)?;
        self.prepare_key_value(
            key_value,
            batch_dim,
            state.state_type.physical_prefix_length(),
            precalculated_rope,
            state.keys.as_mut(),
            state.values.as_mut(),
            encoder,
        )?;
        state.append_full(batch_dim);
        Ok(())
    }

    fn encode_draft(
        &self,
        query: &mut Allocation<B>,
        key_value: &mut Allocation<B>,
        batch_dim: usize,
        precalculated_rope: &PrecalculatedRoPE<B>,
        state: &mut AttentionState<B>,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        state.prepare(state.cur_context_length, batch_dim, context)?;
        let queries = self.prepare_query(query, batch_dim, precalculated_rope, encoder)?;
        self.prepare_key_value(
            key_value,
            batch_dim,
            state.state_type.physical_prefix_length(),
            precalculated_rope,
            state.keys.as_mut(),
            state.values.as_mut(),
            encoder,
        )?;
        self.attention_core.encode(
            AttentionCoreEncodeArguments {
                queries: &queries,
                keys: state.keys.as_ref(),
                values: state.values.as_ref(),
                suffix_length: batch_dim,
                trie: None,
                sinks: None,
                state_type: &state.state_type,
            },
            encoder,
        )
    }

    fn prepare_query(
        &self,
        query: &mut Allocation<B>,
        batch_dim: usize,
        precalculated_rope: &PrecalculatedRoPE<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut queries =
            encoder.allocate_scratch(size_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type))?;
        self.query_prepare.encode(
            &*query,
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

    fn prepare_key_value(
        &self,
        key_value: &mut Allocation<B>,
        batch_dim: usize,
        key_value_offset: usize,
        precalculated_rope: &PrecalculatedRoPE<B>,
        keys: &mut dyn Buffer<Backend = B>,
        values: &mut dyn Buffer<Backend = B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let mut unused_queries = encoder.allocate_scratch(self.data_type.size_in_bytes())?;
        self.key_value_prepare.encode(
            &*key_value,
            &mut unused_queries,
            Some(keys),
            Some(values),
            Some(&precalculated_rope.cosines),
            Some(&precalculated_rope.sines),
            0,
            Some(self.num_kv_heads as u32),
            self.head_dim as u32,
            Some(precalculated_rope.dim as u32),
            Some(key_value_offset as u32),
            batch_dim as u32,
            encoder,
        );
        Ok(())
    }
}
