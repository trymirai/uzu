//! RMS Normalization encodable.

use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
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
    ParameterError(ParameterLoaderError<B>),
}

pub struct RMSNorm<B: Backend> {
    kernel: <B::Kernels as Kernels>::RMSNormKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: Rc<RefCell<B::Buffer>>,
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
        let scales = parameter_tree.leaf_array("scales").map_err(RMSNormError::ParameterError)?;

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
            input_array_id == output_array_id,
        )
        .map_err(RMSNormError::BackendError)?;

        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
            scales_buffer: scales.buffer(),
            use_sampling_range: false,
        })
    }

    /// When enabled, this RMSNorm only runs on `state.sampling_start()..+state.sampling_length()`.
    /// This is useful for the final output norm before readout/sampling in prefill.
    pub fn with_sampling_range(mut self) -> Self {
        self.use_sampling_range = true;
        self
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let input_array = state.array(self.input_array_id);
        let output_array = state.array(self.output_array_id);

        let suffix_length = input_array.shape()[0];
        let element_count = input_array.shape()[1];

        let input_elem_size = input_array.data_type().size_in_bytes();
        let output_elem_size = output_array.data_type().size_in_bytes();

        let (batch_start, batch_len) = if self.use_sampling_range {
            (state.sampling_start(), state.sampling_length())
        } else {
            (0, state.active_row_count())
        };

        let batch_len = batch_len.min(suffix_length.saturating_sub(batch_start));
        if batch_len == 0 {
            return Ok(());
        }

        let row_size_in_bytes = element_count * input_elem_size;
        let input_offset = batch_start * row_size_in_bytes;

        let output_row_size_in_bytes = element_count * output_elem_size;
        let output_offset = batch_start * output_row_size_in_bytes;

        let input_buffer = (self.input_array_id != self.output_array_id).then(|| input_array.buffer());
        let input_buffer_borrow = input_buffer.as_ref().map(|b| b.borrow());

        self.kernel.encode(
            input_buffer_borrow.as_deref().map(|b| (b, input_offset)),
            self.scales_buffer.borrow().deref(),
            (output_array.buffer().borrow_mut().deref_mut(), output_offset),
            batch_len as u32,
            element_count as u32,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            self.config.upcast_mode == UpcastMode::FullLayer,
            encoder,
        );
        Ok(())
    }
}
