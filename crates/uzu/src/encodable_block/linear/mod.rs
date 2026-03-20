mod full_precision;
mod quantized;

pub use full_precision::{FullPrecisionLinear, FullPrecisionLinearError};
pub use quantized::{QuantizedLinear, QuantizedLinearError};
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer},
    config::LinearConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
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
        }
    }
}
