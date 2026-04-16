mod full_precision;
mod qlora_wrapper;
mod quantized;

pub use full_precision::{FullPrecisionLinear, FullPrecisionLinearError};
pub use qlora_wrapper::{QLoRALinearWrapper, QLoRALinearWrapperError};
pub use quantized::{QuantizedLinear, QuantizedLinearError};
use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::LinearConfig,
    parameters::ParameterTree,
};

pub trait Linear<B: Backend> {
    fn encode(
        &self,
        context: &B::Context,
        input: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
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
                )?;
                Ok(Box::new(block))
            },
        }
    }
}
