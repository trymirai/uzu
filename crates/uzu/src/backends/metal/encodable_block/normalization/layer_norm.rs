//! LayerNorm encodable.

use std::rc::Rc;

use crate::backends::metal::{ProtocolObject,
    Buffer, ComputeCommandEncoderRef, MTLCommandBuffer,
    MTLCommandEncoder, MTLDeviceExt, MTLResourceOptions,
};

use super::super::{EncodableBlock, EncodingParameters};
use crate::{
    Array, DataType,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::layer_norm::{
            LayerNormArguments, LayerNormError, LayerNormKernel,
        },
    },
    config::{NormalizationConfig, UpcastMode},
    parameters::ParameterTree,
};

pub struct LayerNorm {
    kernel: LayerNormKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: Buffer,
}

impl LayerNorm {
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, LayerNormError> {
        // Load scales from parameter tree
        let scales_param = parameter_tree.leaf("scales").map_err(|e| {
            LayerNormError::MetalError(MTLError::Library(
                crate::backends::metal::error::LibraryError::Custom(format!(
                    "Failed to load scales: {:?}",
                    e
                )),
            ))
        })?;

        let scales_data = scales_param.buffer();
        let scales_buffer = context.device.new_buffer_with_data(
            scales_data,
            MTLResourceOptions::STORAGE_MODE_SHARED,
        ).expect("Failed to create scales buffer");

        let accumulation_data_type: DataType =
            config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let (input_type, scales_type, output_type) = match config.upcast_mode {
            UpcastMode::OnlyNormalization => {
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
            UpcastMode::FullLayer => {
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
        };

        let kernel = LayerNormKernel::new(
            context,
            input_type,
            scales_type,
            output_type,
            accumulation_data_type,
            config.upcast_mode == UpcastMode::FullLayer,
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

impl EncodableBlock for LayerNorm {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        parameters: &EncodingParameters,
    ) {
        let compute_encoder = command_buffer.new_compute_command_encoder()
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
        compute_encoder: ComputeCommandEncoderRef<'_>,
        _parameters: &EncodingParameters,
    ) {
        let input_binding = state.arrays(&[self.input_array_id]);
        let output_binding = state.arrays(&[self.output_array_id]);

        let input_shape = {
            let input_array = input_binding[0].borrow();
            input_array.shape().to_vec()
        };

        let mut input_array = input_binding[0].borrow_mut();
        let mut output_array = output_binding[0].borrow_mut();

        let input_buffer = unsafe { input_array.mtl_buffer() };
        let output_buffer = unsafe { output_array.mtl_buffer() };

        let batch_size = input_shape[0] as i32;
        let model_dim = input_shape[1] as i32;

        if let Err(e) = self.kernel.encode(
            compute_encoder,
            LayerNormArguments {
                input_buffer: &input_buffer,
                scales_buffer: &self.scales_buffer,
                output_buffer: &output_buffer,
                batch_size,
                model_dim,
                epsilon: self.config.epsilon,
                scale_offset: self.config.scale_offset.unwrap_or(0.0),
            },
        ) {
            eprintln!("Failed to encode LayerNorm kernel: {:?}", e);
        }
    }
}
