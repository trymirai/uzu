mod full_precision;
mod qlora_wrapper;
mod quantized;

pub use full_precision::{FullPrecisionLinear, FullPrecisionLinearError};
pub use qlora_wrapper::{QLoRALinearWrapper, QLoRALinearWrapperError};
pub use quantized::{QuantizedLinear, QuantizedLinearError};
use thiserror::Error;

use crate::{
    backends::common::{Backend, Encoder},
    config::LinearConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub trait Linear<B: Backend> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error>;
}

#[derive(Debug, Error)]
pub enum LinearBlockError<B: Backend> {
    #[error("QuantizedLinear error: {0}")]
    QuantizedLinearError(#[from] QuantizedLinearError<B>),
    #[error("FullPrecisionLinear error: {0}")]
    FullPrecisionLinearError(#[from] FullPrecisionLinearError<B>),
    #[error("QLoRALinearWrapper error: {0}")]
    QLoRALinearWrapperError(#[from] QLoRALinearWrapperError<B>),
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
                )?;
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
                )?;
                Ok(Box::new(block))
            },
            LinearConfig::QLoRA {
                quantization,
                lora_rank,
                lora_scale,
            } => {
                let block = QLoRALinearWrapper::new(
                    context,
                    quantization,
                    *lora_rank,
                    *lora_scale,
                    input_dimension,
                    output_dimension_sum,
                    parameter_tree,
                    input_array_id,
                    output_array_id,
                )?;
                Ok(Box::new(block))
            },
        }
    }
}
