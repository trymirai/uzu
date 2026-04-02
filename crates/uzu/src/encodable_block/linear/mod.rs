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
    backends::common::{Backend, Encoder},
    config::LinearConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
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
                    None,
                    input_array_id,
                    output_array_id,
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
                    input_array_id,
                    output_array_id,
                )?;
                Ok(Box::new(block))
            },
        }
    }

    pub fn new_with_output_hadamard(
        context: &B::Context,
        config: &LinearConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<B::Context>,
        output_factors: B::Buffer,
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
                input_array_id,
                output_array_id,
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
                input_array_id,
                output_array_id,
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
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::Buffer>), LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        match config {
            LinearConfig::RHTLinearWrapper {
                inner_config,
                ..
            } => {
                let input_factors = parameter_tree.leaf("input_factors")?.read_buffer()?;
                let output_factors = parameter_tree.leaf("output_factors")?.read_buffer()?;
                let inner_tree = parameter_tree.subtree("inner_linear")?;
                let inner_linear = Self::new_with_output_hadamard(
                    context,
                    inner_config,
                    input_array_id,
                    output_array_id,
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
                    input_array_id,
                    output_array_id,
                )?;
                Ok((linear, None))
            },
        }
    }
}
