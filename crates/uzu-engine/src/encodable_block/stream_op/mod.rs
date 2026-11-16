use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer},
    config::{residual::AnyResidualConfig, stream_op::AnyStreamOpConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding_table_disk::EmbeddingTableDiskError,
        linear::{LinearBlockError, LinearEncodeError},
        normalization::NormalizationNewError,
        stream_op::{
            ngram_injection::{NGramInjection, NGramInjectionState},
            per_layer_embedding_injection::PerLayerEmbeddingInjection,
            residual_scale::ResidualScale,
        },
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

pub mod ngram_injection;
pub mod per_layer_embedding_injection;
pub mod residual_scale;

#[derive(Debug, Error)]
pub enum StreamOpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Embedding table error: {0}")]
    EmbeddingTable(#[from] EmbeddingTableDiskError<B>),
    #[error("N-gram injection requires a gated residual")]
    NGramInjectionRequiresGatedResidual,
    #[error("{0} requires an additive residual")]
    RequiresAdditiveResidual(&'static str),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

#[derive(Debug, Error)]
pub enum StreamOpEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
    #[error("Per-layer embedding injection requires per-layer inputs")]
    MissingPerLayerInputs,
    #[error("Stateful stream op requires state")]
    MissingState,
}

pub enum StreamOpState<B: Backend> {
    NGramInjection(NGramInjectionState<B>),
}

impl<B: Backend> StreamOpState<B> {
    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        match self {
            Self::NGramInjection(state) => state.encode_accept(accepted_indices, command_buffer),
        }
    }

    pub fn as_ngram_injection_mut(&mut self) -> Option<&mut NGramInjectionState<B>> {
        match self {
            Self::NGramInjection(state) => Some(state),
        }
    }
}

pub enum StreamOp<B: Backend> {
    NGramInjection(NGramInjection<B>),
    PerLayerEmbeddingInjection(PerLayerEmbeddingInjection<B>),
    ResidualScale(ResidualScale<B>),
}

impl<B: Backend> StreamOp<B> {
    pub async fn new(
        model_dim: u32,
        layer_index: u32,
        num_layers: u32,
        data_type: DataType,
        config: &AnyStreamOpConfig,
        residual_config: &AnyResidualConfig,
        parameter_tree: ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, StreamOpNewError<B>> {
        match config {
            AnyStreamOpConfig::NGramInjectionConfig(config) => {
                let AnyResidualConfig::GatedResidualConfig(residual_config) = residual_config else {
                    return Err(StreamOpNewError::NGramInjectionRequiresGatedResidual);
                };
                Ok(Self::NGramInjection(
                    NGramInjection::new(model_dim, data_type, config, residual_config, &parameter_tree, context)
                        .await?,
                ))
            },
            AnyStreamOpConfig::PerLayerEmbeddingInjectionConfig(_) | AnyStreamOpConfig::ResidualScaleConfig(_)
                if !matches!(residual_config, AnyResidualConfig::AdditiveResidualConfig(_)) =>
            {
                Err(StreamOpNewError::RequiresAdditiveResidual(match config {
                    AnyStreamOpConfig::PerLayerEmbeddingInjectionConfig(_) => "Per-layer embedding injection",
                    _ => "Residual scale",
                }))
            },
            AnyStreamOpConfig::PerLayerEmbeddingInjectionConfig(config) => Ok(Self::PerLayerEmbeddingInjection(
                PerLayerEmbeddingInjection::new(
                    model_dim,
                    layer_index,
                    num_layers,
                    data_type,
                    config,
                    &parameter_tree,
                    context,
                )
                .await?,
            )),
            AnyStreamOpConfig::ResidualScaleConfig(_) => {
                Ok(Self::ResidualScale(ResidualScale::new(model_dim, data_type, &parameter_tree, context).await?))
            },
        }
    }

    pub fn create_empty_state(
        &self,
        context: &B::Context,
    ) -> Result<Option<StreamOpState<B>>, B::Error> {
        match self {
            Self::NGramInjection(op) => Ok(Some(StreamOpState::NGramInjection(op.create_empty_state(context)?))),
            Self::PerLayerEmbeddingInjection(_) | Self::ResidualScale(_) => Ok(None),
        }
    }

    pub fn as_ngram_injection(&self) -> Option<&NGramInjection<B>> {
        match self {
            Self::NGramInjection(op) => Some(op),
            Self::PerLayerEmbeddingInjection(_) | Self::ResidualScale(_) => None,
        }
    }

    pub fn speculation_supported(&self) -> bool {
        match self {
            Self::NGramInjection(_) | Self::PerLayerEmbeddingInjection(_) | Self::ResidualScale(_) => true,
        }
    }

    pub fn encode(
        &self,
        stream: B::ScratchBuffer,
        batch_dim: &BatchTopology<B>,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        state: Option<&mut StreamOpState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, StreamOpEncodeError<B>> {
        let row_count = batch_dim.size();
        match self {
            Self::NGramInjection(op) => {
                let Some(StreamOpState::NGramInjection(state)) = state else {
                    // technically, lalamo supports stateless ngram embeddings
                    // by essentially throwing out its work each time
                    return Err(StreamOpEncodeError::MissingState);
                };
                op.encode(stream, batch_dim, state, command_buffer)
            },
            Self::PerLayerEmbeddingInjection(op) => {
                let per_layer_inputs = per_layer_inputs.ok_or(StreamOpEncodeError::MissingPerLayerInputs)?;
                op.encode(stream, per_layer_inputs, row_count, command_buffer)
            },
            Self::ResidualScale(op) => Ok(op.encode(stream, row_count, command_buffer)),
        }
    }
}
