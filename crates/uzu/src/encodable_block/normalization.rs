//! Unified normalization encodables.

use thiserror::Error;

use super::{LayerNorm, LayerNormError, RMSNorm, RMSNormError};
use crate::{
    DataType,
    backends::common::{Backend, Encoder},
    config::NormalizationConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

#[derive(Debug, Error)]
pub enum NormalizationError<B: Backend> {
    #[error("LayerNorm error: {0}")]
    LayerNormError(#[from] LayerNormError<B>),
    #[error("RMSNorm error: {0}")]
    RMSNormError(#[from] RMSNormError<B>),
}

/// Unified normalization encodable that can be either LayerNorm or RMSNorm.
pub enum Normalization<B: Backend> {
    LayerNorm(LayerNorm<B>),
    RMSNorm(RMSNorm<B>),
}

impl<B: Backend> Normalization<B> {
    /// Creates the appropriate normalization encodable based on config.
    /// Uses LayerNorm if subtract_mean is true, otherwise RMSNorm.
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, NormalizationError<B>> {
        if config.subtract_mean {
            Ok(Self::LayerNorm(LayerNorm::new(
                context,
                intermediate_data_type,
                config,
                input_array_id,
                output_array_id,
                parameter_tree,
            )?))
        } else {
            Ok(Self::RMSNorm(RMSNorm::new(
                context,
                intermediate_data_type,
                config,
                input_array_id,
                output_array_id,
                parameter_tree,
                None,
                false,
            )?))
        }
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        match self {
            Self::LayerNorm(layer_norm) => layer_norm.encode(state, encoder),
            Self::RMSNorm(rms_norm) => rms_norm.encode(state, encoder),
        }
    }
}
