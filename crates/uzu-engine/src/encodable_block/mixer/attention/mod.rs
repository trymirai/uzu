use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{AttentionKernel, AttentionKernelConfig, AttentionPrepareKernel, SigmoidGateKernel},
    },
    config::{rope::AnyRoPEConfig, token_mixer::attention::AttentionConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{
            Mixer, MixerState,
            attention::{
                mode::LinearProjection,
                qkv_norm::{QKVNorm, QKVNormError},
                rope::PrecalculatedRoPE,
            },
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

mod mode;
mod qkv_norm;
mod state;

pub use state::{ATTENTION_SUFFIX_CAPACITY, AttentionState, KVCacheView};

pub mod rope;

pub struct Attention<B: Backend> {
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: Option<u32>,
    ring_capacity: Option<u32>,
    max_rope_length: Option<u32>,
    data_type: DataType,
    projection_dim: u32,
    qkv: LinearProjection<B>,
    prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    sinks: Option<Allocation<B>>,
    kernel: <B::Kernels as Kernels>::AttentionKernel,
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
        hidden_dim: u32,
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

        let qkv_projection_tree = parameter_tree.subtree("qkv_projection");
        let qkv_dim = if let Some(num_kv_heads) = num_kv_heads {
            let kv_dim = num_kv_heads * head_dim;
            q_dim + kv_dim + kv_dim
        } else {
            q_dim
        };
        let projection_dim = if config.has_gate {
            qkv_dim + q_dim
        } else {
            qkv_dim
        };
        let (qkv_projection, in_projection_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
            hidden_dim,
            [projection_dim],
            config.has_qkv_biases,
            context,
            data_type,
            &qkv_projection_tree,
        )?;

        let query_norm_config = config.query_norm_config.clone();
        // TODO: Fix lalamo config, those two must be None if kv sharing.
        let key_norm_config = (!is_kv_sharing).then(|| config.key_norm_config.clone()).flatten();
        let value_norm_config = (!is_kv_sharing).then(|| config.value_norm_config()).flatten();
        let packed_projection_heads = projection_dim / head_dim;
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
                    packed_projection_heads,
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

        assert!(sliding_window_size.is_none_or(|size| size > 0), "zero sliding window size");
        let ring_capacity = sliding_window_size;
        let is_kv_cache_ring = ring_capacity.is_some();

        let kernel = <B::Kernels as Kernels>::AttentionKernel::new(
            context,
            AttentionKernelConfig {
                head_dim,
                num_groups,
                num_q_heads,
                has_sinks: sinks.is_some(),
                is_kv_cache_ring,
                is_causal,
                sliding_window_size,
                scale: config.scale,
                data_type,
            },
        )
        .map_err(AttentionNewError::Backend)?;

        let gate_kernel = config
            .has_gate
            .then(|| <B::Kernels as Kernels>::SigmoidGateKernel::new(context, data_type))
            .transpose()
            .map_err(AttentionNewError::Backend)?;

        let out_projection = <dyn Linear<B>>::new(
            q_dim,
            [hidden_dim],
            config.has_out_biases,
            context,
            data_type,
            &parameter_tree.subtree("out_projection"),
        )?;

        Ok((
            Self {
                head_dim,
                num_q_heads,
                num_kv_heads,
                ring_capacity,
                max_rope_length,
                data_type,
                projection_dim,
                qkv: LinearProjection {
                    lin: qkv_projection,
                    norm: qkv_norm,
                },
                prepare,
                sinks,
                kernel,
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

    fn max_context_length(&self) -> Option<u32> {
        self.max_rope_length
    }

    fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
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
        encoder.push_debug_group("attention");

        assert_eq!(precalculated_rope.is_some(), self.max_rope_length.is_some(), "precalculated rope mismatch");

        let state =
            state.map(|state| state.downcast::<AttentionState<B>>().expect("incorrect type of attention state"));
        let output = self.attend(hidden, precalculated_rope, batch_dim, state, encoder)?;

        encoder.pop_debug_group();

        Ok(output)
    }
}
