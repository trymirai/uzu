use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer,
        kernel::{Kernels, RMSNormHadamardMulKernel},
    },
    config::{NormalizationConfig, UpcastMode},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum RMSNormHadamardError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
}

pub struct RMSNormHadamard<B: Backend> {
    kernel: <B::Kernels as Kernels>::RMSNormHadamardMulKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: Rc<RefCell<B::Buffer>>,
    pub(crate) hadamard_factors_buffer: Rc<RefCell<B::Buffer>>,
}

impl<B: Backend> RMSNormHadamard<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        norm_parameter_tree: &ParameterTree<B::Context>,
        hadamard_factors_buffer: Rc<RefCell<B::Buffer>>,
    ) -> Result<Self, RMSNormHadamardError<B>> {
        let scales = norm_parameter_tree.leaf_array("scales").map_err(RMSNormHadamardError::ParameterError)?;

        let accumulation_data_type: DataType = config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let (input_type, scales_type, output_type) = match config.upcast_mode {
            UpcastMode::OnlyNormalization => (intermediate_data_type, scale_data_type, scale_data_type),
            UpcastMode::FullLayer => (intermediate_data_type, scale_data_type, scale_data_type),
        };

        let kernel = <B::Kernels as Kernels>::RMSNormHadamardMulKernel::new(
            context,
            input_type,
            scales_type,
            output_type,
            accumulation_data_type,
            input_array_id == output_array_id,
        )
        .map_err(RMSNormHadamardError::BackendError)?;

        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
            scales_buffer: scales.buffer(),
            hadamard_factors_buffer,
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let input_binding = state.arrays(&[self.input_array_id]);
        let output_binding = state.arrays(&[self.output_array_id]);

        let input_shape = {
            let input_array = input_binding[0].borrow();
            input_array.shape().to_vec()
        };

        let input_array = input_binding[0].borrow();
        let output_array = output_binding[0].borrow_mut();

        let suffix_length = input_shape[0];
        let input_elem_size = input_array.data_type().size_in_bytes();
        let output_elem_size = output_array.data_type().size_in_bytes();

        let (batch_start, batch_len) = (0, state.active_suffix_length());
        let batch_len = batch_len.min(suffix_length.saturating_sub(batch_start));
        if batch_len == 0 {
            return Ok(());
        }

        let row_size_in_bytes = input_shape[1] * input_elem_size;
        let input_offset = batch_start * row_size_in_bytes;

        let output_row_size_in_bytes = input_shape[1] * output_elem_size;
        let output_offset = batch_start * output_row_size_in_bytes;

        let input_buffer = (self.input_array_id != self.output_array_id).then(|| input_array.buffer());
        let input_buffer_borrow = input_buffer.as_ref().map(|b| b.borrow());

        self.kernel.encode(
            input_buffer_borrow.as_deref().map(|b| (b, input_offset)),
            self.scales_buffer.borrow().deref(),
            (output_array.buffer().borrow_mut().deref_mut(), output_offset),
            self.hadamard_factors_buffer.borrow().deref(),
            batch_len as u32,
            input_shape[1] as u32,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            self.config.upcast_mode == UpcastMode::FullLayer,
            command_buffer,
        );
        Ok(())
    }
}
