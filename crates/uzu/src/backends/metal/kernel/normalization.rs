use std::rc::Rc;

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    LayerNormError, LayerNormKernelEncodable, RMSNormError,
    RMSNormKernelEncodable,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
    config::NormalizationConfig,
    parameters::ParameterTree,
};

/// Unified error type for normalization operations.
#[derive(Debug, thiserror::Error)]
pub enum NormalizationError {
    #[error("LayerNorm error: {0}")]
    LayerNorm(#[from] LayerNormError),
    #[error("RMSNorm error: {0}")]
    RMSNorm(#[from] RMSNormError),
}

/// Unified normalization encodable that can be either LayerNorm or RMSNorm.
pub enum NormalizationEncodable {
    LayerNorm(LayerNormKernelEncodable),
    RMSNorm(RMSNormKernelEncodable),
}

impl NormalizationEncodable {
    /// Creates the appropriate normalization encodable based on config.
    /// Uses LayerNorm if subtract_mean is true, otherwise RMSNorm.
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, NormalizationError> {
        if config.subtract_mean {
            // Use LayerNorm (subtract mean before normalization)
            Ok(NormalizationEncodable::LayerNorm(
                LayerNormKernelEncodable::new(
                    context,
                    intermediate_data_type,
                    config,
                    input_array_id,
                    output_array_id,
                    parameter_tree,
                )?,
            ))
        } else {
            // Use RMSNorm (no mean subtraction)
            Ok(NormalizationEncodable::RMSNorm(RMSNormKernelEncodable::new(
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

impl EncodableWithState for NormalizationEncodable {
    fn encode(
        &self,
        state: &mut dyn ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        match self {
            NormalizationEncodable::LayerNorm(layer_norm) => {
                layer_norm.encode(state, command_buffer, parameters)
            },
            NormalizationEncodable::RMSNorm(rms_norm) => {
                rms_norm.encode(state, command_buffer, parameters)
            },
        }
    }
}
