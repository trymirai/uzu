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
    backends::common::{Backend, Encoder, Kernels, kernel::HadamardTransformKernel},
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
        Self::new_with_output_hadamard_and_flags(
            context,
            config,
            input_array_id,
            output_array_id,
            parameter_tree,
            output_factors,
            input_dim,
            output_dim,
            false,
        )
    }

    fn new_with_output_hadamard_and_flags(
        context: &B::Context,
        config: &LinearConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<B::Context>,
        output_factors: B::Buffer,
        input_dim: usize,
        output_dim: usize,
        rms_norm_fuses_a_down: bool,
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
                None,
            )?)),
            LinearConfig::QLoRA {
                quantization,
                lora_rank,
                lora_scale,
            } => {
                if rms_norm_fuses_a_down {
                    Ok(Box::new(QLoRALinearWrapper::new_with_rms_norm_fused_a_down(
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
                    )?))
                } else {
                    Ok(Box::new(QLoRALinearWrapper::new(
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
                    )?))
                }
            },
            inner_config => unimplemented!("{inner_config:?} doesn't support fused output hadamard"),
        }
    }

    /// Like `new_extracting_input_hadamard` but also returns an offline-composed
    /// `adapter_down_prime = A_down @ H` buffer when the inner linear is a QLoRA variant.
    ///
    /// The third element is `Some(adapter_down_prime)` when:
    /// - config is `RHTLinearWrapper { inner_config: QLoRA { .. } }` AND
    /// - the H application at load time succeeds.
    ///
    /// Callers should pass `adapter_down_prime` to `RMSNorm::new_with_lora_fuse` so
    /// the A_down GEMV is eliminated from the decode dispatch path.
    pub fn new_extracting_input_hadamard<const N: usize>(
        config: &LinearConfig,
        _has_biases: bool,
        input_dimension: usize,
        output_dimensions: [usize; N],
        context: &B::Context,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::Buffer>, Option<B::Buffer>), LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        match config {
            LinearConfig::RHTLinearWrapper {
                inner_config,
                ..
            } => {
                let input_factors = parameter_tree.leaf("input_factors")?.read_buffer()?;
                let output_factors = parameter_tree.leaf("output_factors")?.read_buffer()?;
                let inner_tree = parameter_tree.subtree("inner_linear")?;

                // Compute adapter_down_prime = H(A_down rows) at load time when inner is QLoRA.
                // This enables the fused RMSNorm+A_down decode path (no separate A_down dispatch).
                let adapter_down_prime = if let LinearConfig::QLoRA {
                    lora_rank,
                    ..
                } = inner_config.as_ref()
                {
                    let lora_rank = *lora_rank;
                    let down_weights_leaf = inner_tree.leaf("down_weights")?;
                    // Read a copy of A_down to mutate via the Hadamard kernel.
                    let mut adapter_down_prime = down_weights_leaf.read_buffer()?;
                    // Apply H to each row: treat A_down as a [rank, K] batch where
                    // batch_size = rank, hidden_dim = K = input_dimension.
                    let hadamard_kernel = <<B as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
                        context,
                        down_weights_leaf.data_type(),
                    )
                    .map_err(ParameterLoaderError::BackendError)?;
                    let mut encoder = Encoder::<B>::new(context).expect("Failed to create encoder for H composition");
                    hadamard_kernel.encode(
                        &mut adapter_down_prime,
                        &input_factors,
                        input_dimension as u32,
                        lora_rank as u32,
                        &mut encoder,
                    );
                    encoder
                        .end_encoding()
                        .submit()
                        .wait_until_completed()
                        .map_err(ParameterLoaderError::BackendError)?;
                    Some(adapter_down_prime)
                } else {
                    None
                };

                let inner_linear = Self::new_with_output_hadamard_and_flags(
                    context,
                    inner_config,
                    input_array_id,
                    output_array_id,
                    &inner_tree,
                    output_factors,
                    input_dimension,
                    output_dimension_sum,
                    adapter_down_prime.is_some(),
                )?;
                Ok((inner_linear, Some(input_factors), adapter_down_prime))
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
                Ok((linear, None, None))
            },
        }
    }
}
