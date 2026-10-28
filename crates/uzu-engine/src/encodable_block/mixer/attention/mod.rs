use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::{AttentionKernel, AttentionKernelConfig, AttentionPrepareInt8Kernel, AttentionPrepareKernel},
    },
    config::{rope::AnyRoPEConfig, token_mixer::attention::AttentionConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError, LinearEncodeError},
        mixer::{
            Mixer, MixerEncodeError, MixerState,
            attention::{
                mode::LinearProjection,
                qkv_norm::{QKVNorm, QKVNormError},
                rope::PrecalculatedRoPE,
            },
        },
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

pub mod kv_storage;
mod mode;
mod qkv_norm;
mod state;

pub use state::{ATTENTION_SUFFIX_CAPACITY, AttentionState, KVCacheView};

pub mod rope;

pub struct Attention<B: Backend> {
    head_dim: u32,
    num_q_heads: u32,
    num_groups: u32,
    num_kv_heads: Option<u32>,
    ring_capacity: Option<u32>,
    max_rope_length: Option<u32>,
    data_type: DataType,
    has_gate: bool,
    projection_dim: u32,
    projection: LinearProjection<B>,
    prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    prepare_int8: Option<<B::Kernels as Kernels>::AttentionPrepareInt8Kernel>,
    sinks: Option<B::GlobalGpuBuffer>,
    kernel: <B::Kernels as Kernels>::AttentionKernel,
    out_projection: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum AttentionNewError<B: Backend> {
    #[error("Attention kernel error: {0}")]
    Kernel(#[source] <<B::Kernels as Kernels>::AttentionKernel as AttentionKernel>::Error),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("QKVNorm error: {0}")]
    QKVNorm(#[from] QKVNormError<B>),
}

#[derive(Debug, Error)]
pub enum AttentionEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("Attention kernel error: {0}")]
    Kernel(#[source] <<B::Kernels as Kernels>::AttentionKernel as AttentionKernel>::Error),
}

impl<B: Backend> Attention<B> {
    pub async fn new(
        hidden_dim: u32,
        data_type: DataType,
        rope_config: Option<&AnyRoPEConfig>,
        config: &AttentionConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalGpuBuffer>), AttentionNewError<B>> {
        let is_kv_sharing = config.is_kv_sharing;

        let head_dim = config.head_dim;
        let num_groups = config.num_groups;
        let num_q_heads = config.num_heads;
        let num_kv_heads = (!is_kv_sharing).then_some(num_groups);

        let is_causal = config.is_causal;
        let sliding_window_size = config.sliding_window_size;
        let max_rope_length = rope_config.map(|rope_config| *rope_config.max_sequence_length());

        let q_dim = num_q_heads * head_dim;

        let projection_tree = parameter_tree.subtree("qkvg_projection");
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
        let (projection, in_projection_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
            hidden_dim,
            projection_dim,
            data_type,
            config.has_qkvg_biases,
            &projection_tree,
            context,
        )
        .await?;

        let query_norm_config = config.query_norm_config.clone();
        // TODO: Fix lalamo config, those two must be None if kv sharing.
        let key_norm_config = (!is_kv_sharing).then(|| config.key_norm_config.clone()).flatten();
        let value_norm_config = (!is_kv_sharing).then(|| config.value_norm_config()).flatten();
        let qkv_norm = QKVNorm::new(
            num_q_heads,
            num_kv_heads,
            projection_dim,
            head_dim,
            data_type,
            query_norm_config,
            key_norm_config,
            value_norm_config,
            parameter_tree,
            context,
        )
        .await?;

        let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
            context,
            data_type,
            DataType::F32,
            !is_kv_sharing,
            rope_config.is_some(),
        )
        .map_err(AttentionNewError::Backend)?;
        let sinks = if config.has_sinks {
            Some(parameter_tree.read_buffer("sinks", &[num_q_heads], data_type).await?)
        } else {
            None
        };

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
                gate_row_stride: config.has_gate.then_some(projection_dim),
                is_kv_cache_ring,
                is_causal,
                sliding_window_size,
                scale: config.scale,
                data_type,
            },
        )
        .map_err(AttentionNewError::Kernel)?;
        let prepare_int8 = kernel
            .supports_int8()
            .then(|| {
                <B::Kernels as Kernels>::AttentionPrepareInt8Kernel::new(
                    context,
                    data_type,
                    DataType::F32,
                    head_dim,
                    rope_config.is_some(),
                    true,
                    num_groups,
                    !is_kv_sharing,
                )
            })
            .transpose()
            .map_err(AttentionNewError::Backend)?;

        let out_projection = <dyn Linear<B>>::new(
            q_dim,
            hidden_dim,
            data_type,
            config.has_out_biases,
            &parameter_tree.subtree("out_projection"),
            context,
        )
        .await?;

        Ok((
            Self {
                head_dim,
                num_q_heads,
                num_groups,
                num_kv_heads,
                ring_capacity,
                max_rope_length,
                data_type,
                has_gate: config.has_gate,
                projection_dim,
                projection: LinearProjection {
                    lin: projection,
                    norm: qkv_norm,
                },
                prepare,
                prepare_int8,
                sinks,
                kernel,
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
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MixerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("attention");

        assert_eq!(precalculated_rope.is_some(), self.max_rope_length.is_some(), "precalculated rope mismatch");

        let state =
            state.map(|state| state.downcast::<AttentionState<B>>().expect("incorrect type of attention state"));
        let output = self.attend(hidden, precalculated_rope, batch_dim, state, command_buffer)?;

        Ok(output)
    }
}
