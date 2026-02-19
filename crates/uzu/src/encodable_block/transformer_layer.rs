use thiserror::Error;

use super::{
    FullPrecisionEmbeddingLookup, FullPrecisionEmbeddingReadout, FullPrecisionLinear, QuantizedEmbeddingLookup,
    QuantizedEmbeddingReadout, QuantizedLinear,
};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{matmul::MatmulKernels, mlp_gate_act_mul::MlpGateActMulEncodable},
    },
    config::{DecoderConfig, EmbeddingConfig, LinearConfig, MLPConfig},
    encodable_block::{
        EncodableBlock, FullPrecisionLinearError, MlpBlock, MoeBlock, QuantizedLinearError, moe_block::MoeBlockError,
    },
    forward_pass::state::ArrayId,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LayerError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("QuantizedLinear error: {0}")]
    QuantizedLinearError(#[source] QuantizedLinearError<B>),
    #[error("FullPrecisionLinear error: {0}")]
    FullPrecisionLinearError(#[source] FullPrecisionLinearError<B>),
    #[error("MoeBlock error: {0}")]
    MoeBlockError(#[source] MoeBlockError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[source] ParameterLoaderError),
    #[error("QLoRA linear layer not supported")]
    QLoRaNotSupported,
}

pub fn linear_block<const N: usize, B: Backend + 'static>(
    config: &LinearConfig,
    _has_biases: bool,
    input_dimension: usize,
    output_dimensions: [usize; N],
    context: &B::Context,
    parameter_tree: &ParameterTree<B::Context>,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
) -> Result<Box<dyn EncodableBlock<B>>, LayerError<B>>
where
    B::Kernels: MatmulKernels,
{
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
            .map_err(LayerError::QuantizedLinearError)?;
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
            .map_err(LayerError::FullPrecisionLinearError)?;
            Ok(Box::new(block))
        },
        LinearConfig::QLoRA {
            ..
        } => Err(LayerError::QLoRaNotSupported),
    }
}

/// Creates an MLP block using the unfused implementation (separate up, activation, down)
pub fn mlp_block<B: Backend + 'static>(
    config: &MLPConfig,
    model_dimension: usize,
    hidden_dimension: usize,
    context: &B::Context,
    parameter_tree: &ParameterTree<B::Context>,
) -> Result<Box<dyn EncodableBlock<B>>, LayerError<B>>
where
    B::Kernels: MatmulKernels,
{
    if let crate::config::MLPConfig::Dense(dense_config) = config {
        let data_type: DataType = dense_config.linear_config.activation_precision().into();

        // Up projection (outputs 2*hidden_dimension for gate and up)
        let up_projection = linear_block(
            &dense_config.linear_config,
            false,
            model_dimension,
            [2 * hidden_dimension],
            context,
            &parameter_tree.subtree("up_projection").map_err(LayerError::ParameterLoaderError)?,
            ArrayId::Main,
            ArrayId::MlpFusedUp,
        )?;

        // Gate activation + multiply
        let gate_activation =
            MlpGateActMulEncodable::new(context, data_type, dense_config.activation.clone(), hidden_dimension)
                .map_err(LayerError::BackendError)?;

        // Down projection
        let down_projection = linear_block(
            &dense_config.linear_config,
            false,
            hidden_dimension,
            [model_dimension],
            context,
            &parameter_tree.subtree("down_projection").map_err(LayerError::ParameterLoaderError)?,
            ArrayId::MlpHidden,
            ArrayId::Main,
        )?;

        return Ok(Box::new(MlpBlock::new(up_projection, gate_activation, down_projection)));
    }

    if let crate::config::MLPConfig::MixtureOfExperts(mixture_of_experts_config) = config {
        let mixture_of_experts_block =
            MoeBlock::new(context, mixture_of_experts_config, model_dimension, hidden_dimension, parameter_tree)
                .map_err(LayerError::MoeBlockError)?;
        return Ok(Box::new(mixture_of_experts_block));
    }

    unreachable!("Unknown MLP config")
}

pub fn embed_block<B: Backend + 'static>(
    config: &DecoderConfig,
    context: &B::Context,
    parameter_tree: &ParameterTree<B::Context>,
) -> Box<dyn EncodableBlock<B>> {
    match &config.embedding_config {
        EmbeddingConfig::Tied {
            common,
            precision,
        } => {
            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = FullPrecisionEmbeddingLookup::new(
                context,
                (*precision).into(),
                config.vocab_size,
                config.model_dim,
                common.input_scale,
                &embeddings_tree,
            )
            .expect("Failed to create full precision embedding lookup");

            Box::new(block)
        },
        EmbeddingConfig::Untied {
            common,
            precision,
        } => {
            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = FullPrecisionEmbeddingLookup::new(
                context,
                (*precision).into(),
                config.vocab_size,
                config.model_dim,
                common.input_scale,
                &embeddings_tree,
            )
            .expect("Failed to create full precision embedding lookup");

            Box::new(block)
        },
        EmbeddingConfig::MLXSemiQuantizedUntied {
            common,
            activation_precision,
            ..
        } => {
            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = FullPrecisionEmbeddingLookup::new(
                context,
                (*activation_precision).into(),
                config.vocab_size,
                config.model_dim,
                common.input_scale,
                &embeddings_tree,
            )
            .expect("Failed to create full precision embedding lookup");

            Box::new(block)
        },
        EmbeddingConfig::MLXQuantizedUntied {
            group_size,
            embedding_quantization_mode,
            activation_precision,
            ..
        } => {
            let data_type: DataType = (*activation_precision).into();
            let input_scale = config.embedding_config.common().input_scale.unwrap_or(1.0);

            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingLookup::new_untied_input(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                *group_size,
                *embedding_quantization_mode,
                input_scale,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding lookup kernel");

            Box::new(block)
        },
        EmbeddingConfig::QuantizedTied {
            embedding_quantization_mode,
            activation_precision,
            ..
        } => {
            let data_type: DataType = (*activation_precision).into();

            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            // For QuantizedTied, group_size is implicit (per-row quantization), so group_size == model_dim
            let group_size = config.model_dim;
            let input_scale = config.embedding_config.common().input_scale.unwrap_or(1.0);

            let block = QuantizedEmbeddingLookup::new_tied(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                group_size,
                *embedding_quantization_mode,
                input_scale,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding lookup kernel");

            Box::new(block)
        },
        EmbeddingConfig::MLXQuantizedTied {
            group_size,
            embedding_quantization_mode,
            ..
        } => {
            let data_type: DataType = config.output_norm_config.scale_precision.into();
            let input_scale = config.embedding_config.common().input_scale.unwrap_or(1.0);

            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingLookup::new_tied(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                *group_size,
                *embedding_quantization_mode,
                input_scale,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding lookup kernel");

            Box::new(block)
        },
    }
}

pub fn readout_block<B: Backend + 'static>(
    config: &DecoderConfig,
    context: &B::Context,
    parameter_tree: &ParameterTree<B::Context>,
) -> Box<dyn EncodableBlock<B>>
where
    B::Kernels: MatmulKernels,
{
    match &config.embedding_config {
        EmbeddingConfig::Tied {
            precision,
            ..
        } => {
            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = FullPrecisionEmbeddingReadout::new(
                context,
                (*precision).into(),
                config.vocab_size,
                config.model_dim,
                &embeddings_tree,
            )
            .expect("Failed to create full precision embedding readout");

            Box::new(block)
        },
        EmbeddingConfig::Untied {
            precision,
            ..
        } => {
            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = FullPrecisionEmbeddingReadout::new(
                context,
                (*precision).into(),
                config.vocab_size,
                config.model_dim,
                &embeddings_tree,
            )
            .expect("Failed to create full precision embedding readout");

            Box::new(block)
        },
        EmbeddingConfig::MLXSemiQuantizedUntied {
            group_size,
            activation_precision,
            embedding_quantization_mode,
            ..
        } => {
            let data_type: DataType = (*activation_precision).into();
            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingReadout::new_untied_output(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                *group_size,
                *embedding_quantization_mode,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding readout kernel");

            Box::new(block)
        },
        EmbeddingConfig::QuantizedTied {
            embedding_quantization_mode,
            activation_precision,
            ..
        } => {
            let data_type: DataType = (*activation_precision).into();

            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            // For QuantizedTied, group_size is implicit (per-row quantization), so group_size == model_dim
            let group_size = config.model_dim;

            let block = QuantizedEmbeddingReadout::new_tied(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                group_size,
                *embedding_quantization_mode,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding readout kernel");

            Box::new(block)
        },
        EmbeddingConfig::MLXQuantizedTied {
            group_size,
            embedding_quantization_mode,
            ..
        } => {
            let data_type: DataType = config.output_norm_config.scale_precision.into();

            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingReadout::new_tied(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                *group_size,
                *embedding_quantization_mode,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding readout kernel");

            Box::new(block)
        },
        EmbeddingConfig::MLXQuantizedUntied {
            group_size,
            embedding_quantization_mode,
            ..
        } => {
            let data_type: DataType = config.output_norm_config.scale_precision.into();

            let embeddings_tree = parameter_tree.subtree("embedding").expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingReadout::new_untied_output(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                *group_size,
                *embedding_quantization_mode,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding readout kernel");

            Box::new(block)
        },
    }
}
