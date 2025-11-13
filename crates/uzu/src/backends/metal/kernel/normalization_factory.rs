use std::rc::Rc;

use crate::{
    DataType,
    backends::metal::{
        MTLContext,
        forward_pass::{ArrayId, encodable_with_state::EncodableWithState},
        kernel::{LayerNormKernelEncodable, RMSNormKernelEncodable},
    },
    config::NormalizationConfig,
    parameters::ParameterTree,
};

/// Creates the appropriate normalization encodable based on config.
/// Uses LayerNorm if subtract_mean is true, otherwise RMSNorm.
pub fn create_normalization_encodable(
    context: &MTLContext,
    intermediate_data_type: DataType,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Box<dyn EncodableWithState>, Box<dyn std::error::Error>> {
    if config.subtract_mean {
        // Use LayerNorm (subtract mean before normalization)
        Ok(Box::new(
            LayerNormKernelEncodable::new(
                context,
                intermediate_data_type,
                config,
                input_array_id,
                output_array_id,
                parameter_tree,
            )
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
        ))
    } else {
        // Use RMSNorm (no mean subtraction)
        Ok(Box::new(
            RMSNormKernelEncodable::new(
                context,
                intermediate_data_type,
                config,
                input_array_id,
                output_array_id,
                parameter_tree,
            )
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
        ))
    }
}
