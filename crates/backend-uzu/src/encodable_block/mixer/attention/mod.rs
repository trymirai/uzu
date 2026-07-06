use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::trie::TrieNode,
        kernel::{AttentionPrepareKernel, SigmoidGateKernel},
    },
    config::{rope::AnyRoPEConfig, token_mixer::attention::AttentionConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{
            Mixer, MixerState,
            attention::{
                core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments, AttentionCores},
                qkv_norm::{QKVNorm, QKVNormError},
                rope::PrecalculatedRoPE,
                state::{AttentionState, AttentionStateType},
            },
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

pub(crate) mod core;
pub(crate) mod qkv_norm;
pub(crate) mod state;

pub mod rope;

pub struct Attention<B: Backend> {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: Option<usize>,
    is_causal: bool,
    sliding_window_size: Option<usize>,
    max_rope_length: Option<usize>,
    is_kv_sharing: bool,
    data_type: DataType,
    qkv_projection: Box<dyn Linear<B>>,
    gate_projection: Option<Box<dyn Linear<B>>>,
    qkv_norm: Option<QKVNorm<B>>,
    prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    sinks: Option<Allocation<B>>,
    flat_core: AttentionCores<B>,
    trie_core: AttentionCores<B>,
    gate_kernel: Option<<B::Kernels as Kernels>::SigmoidGateKernel>,
    out_projection: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum AttentionNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("QKVNorm error: {0}")]
    QKVNorm(#[from] QKVNormError<B>),
}

impl<B: Backend> Attention<B> {
    pub fn new(
        hidden_dim: usize,
        data_type: DataType,
        rope_config: Option<&AnyRoPEConfig>,
        config: &AttentionConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<Allocation<B>>), AttentionNewError<B>> {
        let is_kv_sharing = config.is_kv_sharing;

        let head_dim = config.head_dim;
        let num_groups = config.num_groups;
        let num_q_heads = config.num_heads;
        let num_kv_heads = (!is_kv_sharing).then_some(num_groups);

        let is_causal = config.is_causal;
        let sliding_window_size = config.sliding_window_size;
        let max_rope_length = rope_config.map(|rope_config| *rope_config.max_sequence_length());

        let q_dim = num_q_heads * head_dim;

        let has_gate = config.gate_projection_config.is_some();

        // TODO: qkv and gate should be fused to be qkvg in lalamo
        let qkv_projection_tree = parameter_tree.subtree("qkv_projection")?;
        let qkv_projection_output_dimension = if let Some(num_kv_heads) = num_kv_heads {
            let kv_dim = num_kv_heads * head_dim;
            q_dim + kv_dim + kv_dim
        } else {
            q_dim
        };
        let (qkv_projection, in_projection_input_hadamard_factors) = if !has_gate {
            <dyn Linear<B>>::new_extracting_input_hadamard(
                hidden_dim,
                [qkv_projection_output_dimension],
                config.has_qkv_biases,
                context,
                data_type,
                &qkv_projection_tree,
            )?
        } else {
            (
                <dyn Linear<B>>::new(
                    hidden_dim,
                    [qkv_projection_output_dimension],
                    config.has_qkv_biases,
                    context,
                    data_type,
                    &qkv_projection_tree,
                )?,
                None,
            )
        };

        let gate_projection = has_gate
            .then(|| {
                <dyn Linear<B>>::new(
                    hidden_dim,
                    [q_dim],
                    false,
                    context,
                    data_type,
                    &parameter_tree.subtree("gate_projection")?,
                )
            })
            .transpose()?;

        let query_norm_config = config.query_norm_config.clone();
        // TODO: Fix lalamo config, those two must be None if kv sharing.
        let key_norm_config = (!is_kv_sharing).then(|| config.key_norm_config.clone()).flatten();
        let value_norm_config = (!is_kv_sharing).then(|| config.value_norm_config()).flatten();
        let qkv_norm = (query_norm_config.is_some() || key_norm_config.is_some() || value_norm_config.is_some())
            .then(|| {
                QKVNorm::new(
                    context,
                    data_type,
                    query_norm_config,
                    key_norm_config,
                    value_norm_config,
                    parameter_tree,
                    config.num_heads,
                    num_kv_heads.unwrap_or(0), // TODO: should take option
                    config.head_dim,
                )
            })
            .transpose()?;

        let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
            context,
            data_type,
            DataType::F32,
            !is_kv_sharing,
            rope_config.is_some(),
        )
        .map_err(AttentionNewError::Backend)?;

        let sinks = config
            .has_sinks
            .then(|| parameter_tree.leaf("sinks")?.validate(&[num_q_heads], data_type)?.read_allocation())
            .transpose()?;

        let flat_core = AttentionCores::new(
            AttentionCoreNewArguments {
                head_dim,
                num_groups,
                num_q_heads,
                has_sinks: sinks.is_some(),
                is_kv_cache_ring: sliding_window_size.is_some(),
                is_causal,
                is_trie: false,
                sliding_window_size,
                scale: config.scale,
                data_type,
            },
            context,
        )
        .map_err(AttentionNewError::Backend)?;

        let trie_core = AttentionCores::new(
            AttentionCoreNewArguments {
                head_dim,
                num_groups,
                num_q_heads,
                has_sinks: sinks.is_some(),
                is_kv_cache_ring: sliding_window_size.is_some(),
                is_causal,
                is_trie: true,
                sliding_window_size,
                scale: config.scale,
                data_type,
            },
            context,
        )
        .map_err(AttentionNewError::Backend)?;

        let gate_kernel = has_gate
            .then(|| <B::Kernels as Kernels>::SigmoidGateKernel::new(context, data_type))
            .transpose()
            .map_err(AttentionNewError::Backend)?;

        let out_projection = <dyn Linear<B>>::new(
            q_dim,
            [hidden_dim],
            config.has_out_biases,
            context,
            data_type,
            &parameter_tree.subtree("out_projection")?,
        )?;

        Ok((
            Self {
                head_dim,
                num_q_heads,
                num_kv_heads,
                is_causal,
                sliding_window_size,
                max_rope_length,
                is_kv_sharing,
                data_type,
                qkv_projection,
                gate_projection,
                qkv_norm,
                prepare,
                sinks,
                flat_core,
                trie_core,
                gate_kernel,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
    }
}

impl<B: Backend> Mixer<B> for Attention<B> {
    fn speculation_supported(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        self.max_rope_length
    }

    fn create_empty_state(
        &self,
        max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        Ok(Box::new(AttentionState::create_empty(self, max_context_length, context)?))
    }

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        assert_eq!(precalculated_rope.is_some(), self.max_rope_length.is_some(), "precalculated rope mismatch");

        let state =
            state.map(|state| state.downcast::<AttentionState<B>>().expect("incorrect type of attention state"));

        // If we have gate we must duplicate input (linear does hadamard in-place). TODO: fix this properly by adding support for not in place input hadamard
        let (hidden, gate) = if let Some(gate_projection) = &self.gate_projection {
            let mut hidden_copy = encoder.allocate_scratch(hidden.size())?;
            encoder.encode_copy(&hidden, .., &mut hidden_copy, ..);
            let gate = gate_projection.encode(hidden, batch_dim.size(), encoder)?;
            (hidden_copy, Some(gate))
        } else {
            (hidden, None)
        };

        let mut qkv = self.qkv_projection.encode(hidden, batch_dim.size(), encoder)?;

        if let Some(qkv_norm) = &self.qkv_norm {
            qkv_norm.encode(&mut qkv, batch_dim.size(), encoder)?;
        }

        let mut queries = encoder
            .allocate_scratch(size_for_shape(&[self.num_q_heads, batch_dim.size(), self.head_dim], self.data_type))?;

        let mut attention_output = if let Some(mut state) = state {
            assert!(matches!(state, MaybeMut::Mut(_)) == !self.is_kv_sharing);

            let (prepare_keys, prepare_values, kv_token_offset) = match &mut state {
                MaybeMut::Const(_) => (None, None, None),
                MaybeMut::Mut(state) => (
                    Some(state.keys.as_mut()),
                    Some(state.values.as_mut()),
                    Some(state.state_type.physical_prefix_length()),
                ),
            };

            self.prepare.encode(
                &qkv,
                &mut queries,
                prepare_keys,
                prepare_values,
                precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
                precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
                self.num_q_heads as u32,
                self.num_kv_heads.map(|num_kv_heads| num_kv_heads as u32),
                self.head_dim as u32,
                precalculated_rope.map(|precalculated_rope| precalculated_rope.dim as u32),
                kv_token_offset.map(|kv_token_offset| kv_token_offset as u32),
                batch_dim.size() as u32,
                encoder,
            );

            let (core, trie) = if batch_dim.is_flat() {
                (&self.flat_core, None)
            } else {
                let mut trie = encoder.allocate_constant(batch_dim.size() * size_of::<TrieNode>())?;
                trie.copyin(batch_dim.nodes());
                (&self.trie_core, Some(trie))
            };

            core.encode(
                AttentionCoreEncodeArguments {
                    queries: &queries,
                    keys: state.keys.as_ref(),
                    values: state.values.as_ref(),
                    suffix_length: batch_dim.size(),
                    trie: trie.as_ref(),
                    sinks: self.sinks.as_ref(),
                    state_type: &state.state_type,
                },
                encoder,
            )?
        } else {
            let Some(num_kv_heads) = self.num_kv_heads else {
                panic!("stateless attention doesn't support kv sharing");
            };

            assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

            let mut keys = encoder
                .allocate_scratch(size_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type))?;
            let mut values = encoder
                .allocate_scratch(size_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type))?;

            self.prepare.encode(
                &qkv,
                &mut queries,
                Some(&mut keys),
                Some(&mut values),
                precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
                precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
                self.num_q_heads as u32,
                self.num_kv_heads.map(|num_kv_heads| num_kv_heads as u32),
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
            )?
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
}
