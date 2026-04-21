mod full_precision;
mod qlora_wrapper;
mod quantized;
mod rht_wrapper;

pub use full_precision::{FullPrecisionLinear, FullPrecisionLinearError};
pub use qlora_wrapper::{QLoRALinearWrapper, QLoRALinearWrapperError};
pub use quantized::{LoraAdapter, QuantizedLinear, QuantizedLinearError};
pub use rht_wrapper::{RHTLinearWrapper, RHTLinearWrapperError};
use thiserror::Error;

use crate::{
    backends::common::{Backend, Buffer, Context, Encoder, Kernels, kernel::HadamardTransformKernel},
    config::LinearConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

/// Pre-composed `A_down @ H` + rank; when `Some`, RMSNorm absorbs the A_down GEMV.
pub struct LoraFusion<B: Backend> {
    pub rotated_adapter_down: B::Buffer,
    pub rank: u32,
}

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
                    false,
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
                rms_norm_fuses_a_down,
            )?)),
            inner_config => unimplemented!("{inner_config:?} doesn't support fused output hadamard"),
        }
    }

    /// Returns input-side work an upstream op can absorb: Hadamard factors and,
    /// for RHT-wrapped QLoRA, the pre-composed `A_down @ H`. Callers without a
    /// fusion site can discard the LoRA slot.
    pub fn new_extracting_input_fusions<const N: usize>(
        config: &LinearConfig,
        _has_biases: bool,
        input_dimension: usize,
        output_dimensions: [usize; N],
        context: &B::Context,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<(Box<dyn Linear<B>>, Option<B::Buffer>, Option<LoraFusion<B>>), LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        match config {
            LinearConfig::RHTLinearWrapper {
                inner_config,
                ..
            } => {
                let input_factors = parameter_tree.leaf("input_factors")?.read_buffer()?;
                let output_factors = parameter_tree.leaf("output_factors")?.read_buffer()?;
                let inner_tree = parameter_tree.subtree("inner_linear")?;

                // Unsupported ranks fail at downstream RMSNormKernel pipeline compile.
                let lora = if let LinearConfig::QLoRA {
                    lora_rank,
                    ..
                } = inner_config.as_ref()
                {
                    Some(LoraFusion {
                        rotated_adapter_down: compose_rotated_adapter_down::<B>(
                            context,
                            &inner_tree,
                            &input_factors,
                            input_dimension,
                            *lora_rank,
                        )?,
                        rank: *lora_rank as u32,
                    })
                } else {
                    None
                };

                let inner_linear = Self::new_with_output_hadamard(
                    context,
                    inner_config,
                    input_array_id,
                    output_array_id,
                    &inner_tree,
                    output_factors,
                    input_dimension,
                    output_dimension_sum,
                    lora.is_some(),
                )?;
                Ok((inner_linear, Some(input_factors), lora))
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

/// Load-time helper for RHT-wrapped QLoRA: reads `A_down`, composes `H(A_down rows)`
/// into a new buffer on-device, then transposes `[rank, K] → [K, rank]` so the
/// fused-RMSNorm inner loop reads contiguous LoRA rows per thread.
fn compose_rotated_adapter_down<B: Backend>(
    context: &B::Context,
    inner_tree: &ParameterTree<B::Context>,
    input_factors: &B::Buffer,
    input_dimension: usize,
    lora_rank: usize,
) -> Result<B::Buffer, LinearBlockError<B>> {
    let down_weights_leaf = inner_tree.leaf("down_weights")?;
    let mut rotated = down_weights_leaf.read_buffer()?;

    let hadamard_kernel =
        <<B as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(context, down_weights_leaf.data_type())
            .map_err(ParameterLoaderError::BackendError)?;
    let mut encoder = Encoder::<B>::new(context).expect("Failed to create encoder for H composition");
    hadamard_kernel.encode(&mut rotated, input_factors, input_dimension as u32, lora_rank as u32, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().map_err(ParameterLoaderError::BackendError)?;

    let elem_size = down_weights_leaf.data_type().size_in_bytes();
    let buf_size = lora_rank * input_dimension * elem_size;
    let transposed = context.create_buffer(buf_size).map_err(ParameterLoaderError::BackendError)?;
    unsafe {
        let src = std::slice::from_raw_parts(rotated.cpu_ptr().as_ptr() as *const u8, buf_size);
        let dst = std::slice::from_raw_parts_mut(transposed.cpu_ptr().as_ptr() as *mut u8, buf_size);
        for r in 0..lora_rank {
            for k in 0..input_dimension {
                let src_byte = (r * input_dimension + k) * elem_size;
                let dst_byte = (k * lora_rank + r) * elem_size;
                dst[dst_byte..dst_byte + elem_size].copy_from_slice(&src[src_byte..src_byte + elem_size]);
            }
        }
    }
    Ok(transposed)
}
