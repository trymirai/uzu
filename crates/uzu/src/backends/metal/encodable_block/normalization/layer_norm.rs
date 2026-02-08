//! LayerNorm encodable.

use super::super::{EncodableBlock, EncodingParameters, Metal};
use crate::{
    DataType,
    backends::{
        common::kernel::LayerNormKernel,
        metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
            MTLComputeCommandEncoder, MTLContext, MTLDeviceExt, MTLError,
            MTLResourceOptions, ProtocolObject, Retained,
            forward_pass::{ArrayId, ForwardPassState},
            kernel::dsl::LayerNormMetalKernel,
        },
    },
    config::{NormalizationConfig, UpcastMode},
    parameters::ParameterTree,
};

pub struct LayerNorm {
    kernel: LayerNormMetalKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl LayerNorm {
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<MTLContext>,
    ) -> Result<Self, MTLError> {
        // Load scales from parameter tree
        let scales_param = parameter_tree.leaf("scales").map_err(|e| {
            MTLError::Library(
                crate::backends::metal::error::LibraryError::Custom(format!(
                    "Failed to load scales: {:?}",
                    e
                )),
            )
        })?;

        let scales_data = scales_param.as_bytes();
        let scales_buffer = context
            .device
            .new_buffer_with_data(
                scales_data,
                MTLResourceOptions::STORAGE_MODE_SHARED,
            )
            .expect("Failed to create scales buffer");

        let accumulation_data_type: DataType =
            config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let kernel = LayerNormMetalKernel::new(
            context,
            intermediate_data_type.into(),
            scale_data_type.into(),
            scale_data_type.into(),
            accumulation_data_type.into(),
        )?;
        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
            scales_buffer,
        })
    }
}

impl EncodableBlock<Metal> for LayerNorm {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters,
    ) {
        let compute_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.encode_with_shared_encoder(state, &compute_encoder, parameters);
        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters,
    ) {
        let input_binding = state.arrays(&[self.input_array_id]);
        let output_binding = state.arrays(&[self.output_array_id]);

        let input_shape = {
            let input_array = input_binding[0].borrow();
            input_array.shape().to_vec()
        };

        let input_array = input_binding[0].borrow_mut();
        let output_array = output_binding[0].borrow_mut();

        let input_buffer = input_array.buffer();
        let output_buffer = output_array.buffer();

        let batch_size = input_shape[0] as u32;
        let model_dim = input_shape[1] as u32;
        let full_layer = if self.config.upcast_mode == UpcastMode::FullLayer {
            1u32
        } else {
            0u32
        };

        self.kernel.encode(
            &input_buffer,
            &self.scales_buffer,
            &output_buffer,
            batch_size,
            model_dim,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            full_layer,
            compute_encoder,
        )
    }
}
