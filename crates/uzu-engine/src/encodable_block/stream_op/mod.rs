use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer},
    config::{residual::AnyResidualConfig, stream_op::AnyStreamOpConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{LinearBlockError, LinearEncodeError},
        normalization::NormalizationNewError,
        stream_op::{
            ngram_injection::NGramInjection, per_layer_embedding_injection::PerLayerEmbeddingInjection,
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
    #[error("N-gram injection requires a gated residual")]
    NGramInjectionRequiresGatedResidual,
    #[error("{0} requires an additive residual")]
    RequiresAdditiveResidual(&'static str),
}

#[derive(Debug, Error)]
pub enum StreamOpEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
    #[error("Per-layer embedding injection requires per-layer inputs")]
    MissingPerLayerInputs,
}

/// Inputs to stream ops other than the stream itself.
// TODO: n-gram injection also needs the token ids and its own state (token history + conv state).
// lalamo rejects stateful stream ops in layers whose mixer has a KV cache, so the state could live
// next to the mixer state in the layer state.
#[derive(Clone, Copy)]
pub struct StreamOpContext<'a, PerLayerInputs> {
    pub batch_dim: &'a BatchTopology,
    pub per_layer_inputs: Option<PerLayerInputs>,
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

    pub fn encode(
        &self,
        stream: B::ScratchBuffer,
        op_context: StreamOpContext<'_, impl BufferRef<Backend = B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, StreamOpEncodeError<B>> {
        let row_count = op_context.batch_dim.size();
        match self {
            Self::NGramInjection(op) => {
                op.encode(stream, op_context.batch_dim, command_buffer).map_err(StreamOpEncodeError::Backend)
            },
            Self::PerLayerEmbeddingInjection(op) => {
                let per_layer_inputs = op_context.per_layer_inputs.ok_or(StreamOpEncodeError::MissingPerLayerInputs)?;
                op.encode(stream, per_layer_inputs, row_count, command_buffer)
            },
            Self::ResidualScale(op) => Ok(op.encode(stream, row_count, command_buffer)),
        }
    }
}
