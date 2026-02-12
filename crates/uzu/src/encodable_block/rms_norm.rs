//! RMS Normalization encodable.

use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{Kernels, RMSNormKernel},
    },
    config::{NormalizationConfig, UpcastMode},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum RMSNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
}

pub struct RMSNorm<B: Backend> {
    kernel: <B::Kernels as Kernels>::RMSNormKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: B::NativeBuffer,
    use_sampling_range: bool,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, RMSNormError<B>> {
        let scales = parameter_tree.leaf("scales").map_err(RMSNormError::ParameterError)?;
        let scales_buffer = scales.buffer().clone();

        let accumulation_data_type: DataType = config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let (input_type, scales_type, output_type) = match config.upcast_mode {
            UpcastMode::OnlyNormalization => (intermediate_data_type, scale_data_type, scale_data_type),
            UpcastMode::FullLayer => (intermediate_data_type, scale_data_type, scale_data_type),
        };

        let kernel = <B::Kernels as Kernels>::RMSNormKernel::new(
            context,
            input_type,
            scales_type,
            output_type,
            accumulation_data_type,
        )
        .map_err(RMSNormError::BackendError)?;

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

impl<B: Backend> EncodableBlock<B> for RMSNorm<B> {
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
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

        self.kernel.encode(
            (input_array.buffer(), input_offset),
            &self.scales_buffer,
            (output_array.buffer(), output_offset),
            batch_len as u32,
            input_shape[1] as u32,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            self.config.upcast_mode == UpcastMode::FullLayer,
            encoder,
        );
    }
}
