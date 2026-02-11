//! RMS Normalization encodable.
use super::super::{EncodableBlock, Metal};
use crate::backends::common::kernel::RMSNormKernel;
use crate::{
    DataType,
    backends::metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLContext, MTLDeviceExt, MTLError,
        MTLResourceOptions, ProtocolObject, Retained,
        kernel::{dsl::RMSNormMetalKernel, rms_norm::RMSNormError},
    },
    config::{NormalizationConfig, UpcastMode},
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub struct RMSNorm {
    kernel: RMSNormMetalKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    use_sampling_range: bool,
}

impl RMSNorm {
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<MTLContext>,
    ) -> Result<Self, RMSNormError> {
        // Load scales from parameter tree
        let scales_param = parameter_tree.leaf("scales").map_err(|e| {
            RMSNormError::MetalError(MTLError::Library(crate::backends::metal::error::LibraryError::Custom(format!(
                "Failed to load scales: {:?}",
                e
            ))))
        })?;

        // TODO: Don't create buffers dynamically, we need to use forward pass storage for thing like this
        let scales_data = scales_param.as_bytes();
        let scales_buffer = context
            .device
            .new_buffer_with_data(scales_data, MTLResourceOptions::STORAGE_MODE_SHARED)
            .expect("Failed to create scales buffer");

        let accumulation_data_type: DataType = config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let (input_type, scales_type, output_type) = match config.upcast_mode {
            UpcastMode::OnlyNormalization => {
                // Input stays as pipeline type, scales stay scale precision, output is scale precision
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
            UpcastMode::FullLayer => {
                // Input stays as pipeline type, scales stay in original precision (will be cast to AccumT inside kernel), output is scale precision
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
        };

        let kernel = RMSNormMetalKernel::new(context, input_type, scales_type, output_type, accumulation_data_type)?;
        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
            scales_buffer,
            use_sampling_range: false,
        })
    }

    /// When enabled, this RMSNorm only runs on `state.sampling_start()..+state.sampling_length()`.
    /// This is useful for the final output norm before readout/sampling in prefill.
    pub fn with_sampling_range(mut self) -> Self {
        self.use_sampling_range = true;
        self
    }
}

impl EncodableBlock<Metal> for RMSNorm {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters<Metal>,
    ) {
        let compute_encoder =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
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
        state: &mut ForwardPassState<Metal>,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters<Metal>,
    ) {
        let input_binding = state.arrays(&[self.input_array_id]);
        let output_binding = state.arrays(&[self.output_array_id]);

        let input_shape = {
            let input_array = input_binding[0].borrow();
            input_array.shape().to_vec()
        };

        let input_array = input_binding[0].borrow_mut();
        let output_array = output_binding[0].borrow_mut();

        let suffix_length = input_shape[0];
        let input_elem_size = input_array.data_type().size_in_bytes();
        let output_elem_size = output_array.data_type().size_in_bytes();
        let input_buffer = input_array.buffer();
        let output_buffer = output_array.buffer();

        let (batch_start, batch_len) = if self.use_sampling_range {
            (state.sampling_start(), state.sampling_length())
        } else {
            (0, state.active_suffix_length())
        };
        let batch_len = batch_len.min(suffix_length.saturating_sub(batch_start));
        if batch_len == 0 {
            return;
        }

        let row_size_in_bytes = input_shape[1] * input_elem_size;
        let input_offset = batch_start * row_size_in_bytes;

        let output_row_size_in_bytes = input_shape[1] * output_elem_size;
        let output_offset = batch_start * output_row_size_in_bytes;

        let batch_size = batch_len as i32;
        let model_dim = input_shape[1] as i32;

        self.kernel.encode(
            (input_buffer, input_offset),
            &self.scales_buffer,
            (output_buffer, output_offset),
            batch_size as u32,
            model_dim as u32,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            self.config.upcast_mode == UpcastMode::FullLayer,
            compute_encoder,
        )
    }
}
