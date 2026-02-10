//! Unified normalization encodables: LayerNorm, RMSNorm, QKNorm.

mod layer_norm;
mod qk_norm;
mod rms_norm;

pub use layer_norm::LayerNorm;
pub use qk_norm::QKNorm;
pub use rms_norm::RMSNorm;

use super::{EncodableBlock, Metal};
use crate::{
    DataType,
    backends::metal::{
        MTLCommandBuffer, MTLComputeCommandEncoder, MTLContext, MTLError, ProtocolObject, Retained,
        kernel::RMSNormError,
    },
    config::NormalizationConfig,
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

/// Unified error type for normalization operations.
#[derive(Debug, thiserror::Error)]
pub enum NormalizationError {
    #[error("LayerNorm error: {0}")]
    LayerNorm(#[from] MTLError),
    #[error("RMSNorm error: {0}")]
    RMSNorm(#[from] RMSNormError),
}

/// Unified normalization encodable that can be either LayerNorm or RMSNorm.
pub enum Normalization {
    LayerNorm(LayerNorm),
    RMSNorm(RMSNorm),
}

impl Normalization {
    /// Creates the appropriate normalization encodable based on config.
    /// Uses LayerNorm if subtract_mean is true, otherwise RMSNorm.
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<MTLContext>,
    ) -> Result<Self, NormalizationError> {
        if config.subtract_mean {
            // Use LayerNorm (subtract mean before normalization)
            Ok(Normalization::LayerNorm(LayerNorm::new(
                context,
                intermediate_data_type,
                config,
                input_array_id,
                output_array_id,
                parameter_tree,
            )?))
        } else {
            // Use RMSNorm (no mean subtraction)
            Ok(Normalization::RMSNorm(RMSNorm::new(
                context,
                intermediate_data_type,
                config,
                input_array_id,
                output_array_id,
                parameter_tree,
            )?))
        }
    }
}

impl EncodableBlock<Metal> for Normalization {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters<Metal>,
    ) {
        match self {
            Normalization::LayerNorm(layer_norm) => layer_norm.encode(state, command_buffer, parameters),
            Normalization::RMSNorm(rms_norm) => rms_norm.encode(state, command_buffer, parameters),
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        match self {
            Normalization::LayerNorm(layer_norm) => layer_norm.supports_shared_encoder(),
            Normalization::RMSNorm(rms_norm) => rms_norm.supports_shared_encoder(),
        }
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        parameters: &EncodingParameters<Metal>,
    ) {
        match self {
            Normalization::LayerNorm(layer_norm) => layer_norm.encode_with_shared_encoder(state, encoder, parameters),
            Normalization::RMSNorm(rms_norm) => rms_norm.encode_with_shared_encoder(state, encoder, parameters),
        }
    }
}
