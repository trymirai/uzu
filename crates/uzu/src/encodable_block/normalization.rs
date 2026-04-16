//! Unified normalization encodables.

use thiserror::Error;

use super::{LayerNorm, LayerNormError, RMSNorm, RMSNormError};
use crate::{
    DataType,
    backends::common::{Allocation, Backend, Encoder},
    config::NormalizationConfig,
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
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, NormalizationError<B>> {
        if config.subtract_mean {
            Ok(Self::LayerNorm(LayerNorm::new(context, intermediate_data_type, config, parameter_tree)?))
        } else {
            Ok(Self::RMSNorm(RMSNorm::new(context, intermediate_data_type, config, parameter_tree, false, false)?))
        }
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        row_offset: usize,
        row_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        match self {
            Self::LayerNorm(layer_norm) => layer_norm.encode(input, row_offset, row_count, encoder),
            Self::RMSNorm(rms_norm) => rms_norm.encode(input, row_offset, row_count, None, encoder),
        }
    }
}
