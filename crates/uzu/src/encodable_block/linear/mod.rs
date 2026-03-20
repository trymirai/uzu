mod full_precision;
mod quantized;
mod rht_wrapper;

pub use full_precision::{FullPrecisionLinear, FullPrecisionLinearError};
pub use quantized::{QuantizedLinear, QuantizedLinearError};
pub use rht_wrapper::{RHTLinearWrapper, RHTLinearWrapperError};
use std::{cell::RefCell, rc::Rc};

use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer},
    config::LinearConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

pub trait Linear<B: Backend> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error>;
}

#[derive(Debug, Error)]
pub enum LinearBlockError<B: Backend> {
    #[error("QuantizedLinear error: {0}")]
    QuantizedLinearError(#[source] QuantizedLinearError<B>),
    #[error("FullPrecisionLinear error: {0}")]
    FullPrecisionLinearError(#[source] FullPrecisionLinearError<B>),
    #[error("RHTLinearWrapper error: {0}")]
    RHTLinearWrapperError(#[source] RHTLinearWrapperError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[source] ParameterLoaderError<B>),
    #[error("QLoRA linear layer not supported")]
    QLoRaNotSupported,
}

impl<B: Backend> dyn Linear<B> {
    pub fn new<const N: usize>(
        config: &LinearConfig,
        _has_biases: bool,
        input_dimension: usize,
        output_dimensions: [usize; N],
        context: &B::Context,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        match config {
            LinearConfig::Quantized(quantization_config) | LinearConfig::MLXQuantized(quantization_config) => {
                let block = QuantizedLinear::new(
                    context,
                    quantization_config,
                    input_dimension,
                    output_dimension_sum,
                    parameter_tree,
                    input_array_id,
                    output_array_id,
                )
                .map_err(LinearBlockError::QuantizedLinearError)?;
                Ok(Box::new(block))
            },
            LinearConfig::FullPrecision {
                precision,
            } => {
                let block = FullPrecisionLinear::new(
                    context,
                    (*precision).into(),
                    input_dimension,
                    output_dimension_sum,
                    parameter_tree,
                    input_array_id,
                    output_array_id,
                )
                .map_err(LinearBlockError::FullPrecisionLinearError)?;
                Ok(Box::new(block))
            },
            LinearConfig::QLoRA {
                ..
            } => Err(LinearBlockError::QLoRaNotSupported),
            LinearConfig::RHTLinearWrapper {
                block_size,
                inner_config,
            } => {
                let block = RHTLinearWrapper::new(
                    context,
                    *block_size,
                    inner_config,
                    input_dimension,
                    output_dimension_sum,
                    parameter_tree,
                    input_array_id,
                    output_array_id,
                )
                .map_err(LinearBlockError::RHTLinearWrapperError)?;
                Ok(Box::new(block))
            },
        }
    }

    pub fn new_extracting_input_hadamard<const N: usize>(
        config: &LinearConfig,
        _has_biases: bool,
        input_dimension: usize,
        output_dimensions: [usize; N],
        context: &B::Context,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<(Box<dyn Linear<B>>, Option<Rc<RefCell<B::Buffer>>>), LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        match config {
            LinearConfig::RHTLinearWrapper {
                block_size,
                inner_config,
            } => {
                let mut block = RHTLinearWrapper::new(
                    context,
                    *block_size,
                    inner_config,
                    input_dimension,
                    output_dimension_sum,
                    parameter_tree,
                    input_array_id,
                    output_array_id,
                )
                .map_err(LinearBlockError::RHTLinearWrapperError)?;
                let factors = block.take_input_hadamard_factors();
                Ok((Box::new(block), factors))
            },
            other => {
                let linear = Self::new(other, _has_biases, input_dimension, output_dimensions, context, parameter_tree, input_array_id, output_array_id)?;
                Ok((linear, None))
            },
        }
    }
}
