mod full_precision;
mod qlora_wrapper;
mod quantized;
mod rht_wrapper;

pub use full_precision::{FullPrecisionLinear, FullPrecisionLinearError};
pub use qlora_wrapper::{QLoRALinearWrapper, QLoRALinearWrapperError};
pub use quantized::{QuantizedLinear, QuantizedLinearError};
pub use rht_wrapper::{RHTLinearWrapper, RHTLinearWrapperError};
use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::LinearConfig,
    parameters::{ParameterLoaderError, ParameterTree},
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
    #[error("RHTLinearWrapper error: {0}")]
    RHTLinearWrapperError(#[from] RHTLinearWrapperError<B>),
    #[error("QLoRALinearWrapper error: {0}")]
    QLoRALinearWrapperError(#[from] QLoRALinearWrapperError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
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
                    None,
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
                    None,
                )?;
                Ok(Box::new(block))
            },
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
                )?;
                Ok(Box::new(block))
            },
        }
    }

    pub fn new_with_output_hadamard(
        context: &B::Context,
        config: &LinearConfig,
        parameter_tree: &ParameterTree<B::Context>,
        output_factors: Allocation<B>,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        match config {
            LinearConfig::Quantized(config) | LinearConfig::MLXQuantized(config) => Ok(Box::new(QuantizedLinear::new(
                context,
                config,
                input_dim,
                output_dim,
                parameter_tree,
                Some(output_factors),
            )?)),
            LinearConfig::QLoRA {
                quantization,
                lora_rank,
                lora_scale,
            } => Ok(Box::new(QLoRALinearWrapper::new(
                context,
                quantization,
                *lora_rank,
                *lora_scale,
                input_dim,
                output_dim,
                parameter_tree,
                Some(output_factors),
            )?)),
            inner_config => unimplemented!("{inner_config:?} doesn't support fused output hadamard"),
        }
    }

    pub fn new_extracting_input_hadamard<const N: usize>(
        config: &LinearConfig,
        _has_biases: bool,
        input_dimension: usize,
        output_dimensions: [usize; N],
        context: &B::Context,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<(Box<dyn Linear<B>>, Option<Allocation<B>>), LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        match config {
            LinearConfig::RHTLinearWrapper {
                inner_config,
                ..
            } => {
                let input_factors = parameter_tree.leaf("input_factors")?.read_allocation()?;
                let output_factors = parameter_tree.leaf("output_factors")?.read_allocation()?;
                let inner_tree = parameter_tree.subtree("inner_linear")?;
                let inner_linear = Self::new_with_output_hadamard(
                    context,
                    inner_config,
                    &inner_tree,
                    output_factors,
                    input_dimension,
                    output_dimension_sum,
                )?;
                Ok((inner_linear, Some(input_factors)))
            },
            other => {
                let linear = Self::new(
                    other,
                    _has_biases,
                    input_dimension,
                    output_dimensions,
                    context,
                    parameter_tree,
                )?;
                Ok((linear, None))
            },
        }
    }
}
