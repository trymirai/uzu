//! LayerNorm encodable.

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
        kernel::{Kernels, LayerNormKernel},
    },
    config::{NormalizationConfig, UpcastMode},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LayerNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
}

pub struct LayerNorm<B: Backend> {
    kernel: <B::Kernels as Kernels>::LayerNormKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: Rc<RefCell<B::Buffer>>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, LayerNormError<B>> {
        let scales = parameter_tree.leaf_array("scales").map_err(LayerNormError::ParameterError)?;

        let accumulation_data_type: DataType = config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let kernel = <B::Kernels as Kernels>::LayerNormKernel::new(
            context,
            intermediate_data_type,
            scale_data_type,
            scale_data_type,
            accumulation_data_type,
            input_array_id == output_array_id,
        )
        .map_err(LayerNormError::BackendError)?;

        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
            scales_buffer: scales.buffer(),
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let input_array = state.array(self.input_array_id);
        let output_array = state.array(self.output_array_id);

        let batch_size = input_array.shape()[0] as u32;
        let model_dim = input_array.shape()[1] as u32;
        let full_layer = if self.config.upcast_mode == UpcastMode::FullLayer {
            1u32
        } else {
            0u32
        };

        let input_buffer = (self.input_array_id != self.output_array_id).then(|| input_array.buffer());
        let input_buffer_borrow = input_buffer.as_ref().map(|b| b.borrow());

        self.kernel.encode(
            input_buffer_borrow.as_deref(),
            self.scales_buffer.borrow().deref(),
            output_array.buffer().borrow_mut().deref_mut(),
            batch_size,
            model_dim,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            full_layer,
            encoder,
        );
        Ok(())
    }
}
