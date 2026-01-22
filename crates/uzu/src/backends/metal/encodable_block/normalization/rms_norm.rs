//! RMS Normalization encodable.

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
        kernel::rms_norm::{
            RMSNormArguments, RMSNormError, RMSNormKernel, RMSNormKernelType,
        },
    },
    config::{NormalizationConfig, UpcastMode},
    parameters::ParameterTree,
};

pub struct RMSNorm {
    kernel: RMSNormKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: Buffer,
    use_sampling_range: bool,
}

impl RMSNorm {
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, RMSNormError> {
        // Load scales from parameter tree
        let scales_param = parameter_tree.leaf("scales").map_err(|e| {
            RMSNormError::MetalError(MTLError::Library(
                crate::backends::metal::error::LibraryError::Custom(format!(
                    "Failed to load scales: {:?}",
                    e
                )),
            ))
        })?;

        // TODO: Don't create buffers dynamically, we need to use forward pass storage for thing like this
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
                // Input stays as pipeline type, scales stay scale precision, output is scale precision
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
            UpcastMode::FullLayer => {
                // Input stays as pipeline type, scales stay in original precision (will be cast to AccumT inside kernel), output is scale precision
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
        };

        let kernel = RMSNormKernel::new_with_mode(
            context,
            input_type,
            scales_type,
            output_type,
            accumulation_data_type,
            RMSNormKernelType::Standard,
            config.upcast_mode == UpcastMode::FullLayer,
        )?;

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

impl EncodableBlock for RMSNorm {
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

        let suffix_length = input_shape[0];
        let input_elem_size = input_array.data_type().size_in_bytes();
        let output_elem_size = output_array.data_type().size_in_bytes();
        let input_buffer = unsafe { input_array.mtl_buffer() };
        let output_buffer = unsafe { output_array.mtl_buffer() };

        let (batch_start, batch_len) = if self.use_sampling_range {
            (state.sampling_start(), state.sampling_length())
        } else {
            (0, state.active_suffix_length())
        };
        let batch_len =
            batch_len.min(suffix_length.saturating_sub(batch_start));
        if batch_len == 0 {
            return;
        }

        let row_size_in_bytes = input_shape[1] * input_elem_size;
        let input_offset = (batch_start * row_size_in_bytes) as u64;

        let output_row_size_in_bytes = input_shape[1] * output_elem_size;
        let output_offset = (batch_start * output_row_size_in_bytes) as u64;

        let batch_size = batch_len as i32;
        let model_dim = input_shape[1] as i32;

        if let Err(e) = self.kernel.encode(
            compute_encoder,
            RMSNormArguments {
                input_buffer: &input_buffer,
                input_offset,
                scales_buffer: &self.scales_buffer,
                output_buffer: &output_buffer,
                output_offset,
                batch_size,
                model_dim,
                epsilon: self.config.epsilon,
                scale_offset: self.config.scale_offset.unwrap_or(0.0),
            },
        ) {
            eprintln!("Failed to encode RMS norm kernel: {:?}", e);
        }
    }
}
