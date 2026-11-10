// XXX: cleanup

use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer},
    config::{residual::AnyResidualConfig, stream_op::AnyStreamOpConfig},
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, stream_op::ngram_injection::NGramInjection},
    safetensors::{ParameterLoaderError, ParameterTree},
};

pub mod ngram_injection;

#[derive(Debug, Error)]
pub enum StreamOpNewError<B: Backend> {
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("N-gram injection requires a gated residual")]
    NGramInjectionRequiresGatedResidual,
}

pub enum StreamOp<B: Backend> {
    NGramInjection(NGramInjection<B>),
}

impl<B: Backend> StreamOp<B> {
    pub async fn new(
        model_dim: u32,
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
            AnyStreamOpConfig::PerLayerEmbeddingInjectionConfig(_) => {
                todo!("per-layer embedding injection as a standalone stream op")
            },
            AnyStreamOpConfig::ResidualScaleConfig(_) => todo!("residual scale as a standalone stream op"),
        }
    }

    // TODO: Inputs other than the stream are undecided: n-gram injection needs the token ids and
    // its own state (token history + conv state), per-layer embedding injection needs the per-layer
    // inputs. Recommended: a `StreamOpContext` (token ids, per-layer inputs) like lalamo's, plus the
    // op's state. lalamo rejects stateful stream ops in layers whose mixer has a KV cache, so the
    // state could live next to the mixer state in the layer state.
    // TODO: called by the layer once the residual add is no longer fused into the norms.
    /// Returns the updated stream. The stream has the layout produced by the residual head's
    /// `expand`.
    #[allow(dead_code)]
    pub fn encode(
        &self,
        stream: B::ScratchBuffer,
        batch_dim: &BatchTopology,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        match self {
            Self::NGramInjection(op) => op.encode(stream, batch_dim, command_buffer),
        }
    }
}
