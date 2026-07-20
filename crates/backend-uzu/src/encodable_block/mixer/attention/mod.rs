use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
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
                core::{AttentionCoreNewArguments, AttentionCores},
                mode::{LinearProjection, QkvProjection},
                qkv_norm::{QKVNorm, QKVNormError},
                rope::PrecalculatedRoPE,
            },
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

pub(crate) mod core;
mod mode;
mod qkv_norm;
mod state;

pub(crate) use state::{ATTENTION_SUFFIX_CAPACITY, AttentionState};

pub mod rope;

pub struct Attention<B: Backend> {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: Option<usize>,
    is_causal: bool,
    sliding_window_size: Option<usize>,
    max_rope_length: Option<usize>,
    data_type: DataType,
    projection: QkvProjection<B>,
    gate_projection: Option<Box<dyn Linear<B>>>,
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
        let projection = QkvProjection::Packed {
            qkv: LinearProjection {
                lin: qkv_projection,
                norm: qkv_norm,
            },
            prepare,
        };

        let sinks = config
            .has_sinks
            .then(|| parameter_tree.leaf("sinks")?.validate(&[num_q_heads], data_type)?.read_allocation())
            .transpose()?;

        let is_kv_cache_ring = is_causal && sliding_window_size.is_some();

        let flat_core = AttentionCores::new(
            AttentionCoreNewArguments {
                head_dim,
                num_groups,
                num_q_heads,
                has_sinks: sinks.is_some(),
                is_kv_cache_ring,
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
                is_kv_cache_ring,
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
                data_type,
                projection,
                gate_projection,
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
        self.attend(hidden, precalculated_rope, batch_dim, state, encoder)
    }
}
