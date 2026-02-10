use std::rc::Rc;

use super::Metal;
use super::{
    EncodableBlock, FullPrecisionEmbeddingLookup, FullPrecisionEmbeddingReadout, FullPrecisionLinear, MlpBlock,
    MlpFusedBlock, MoeBlock, QuantizedEmbeddingLookup, QuantizedEmbeddingReadout, QuantizedLinear,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        kernel::{mlp::MlpGateActMulEncodable, quant_matmul::QuantizationType},
    },
    config::{DecoderConfig, EmbeddingConfig, LinearConfig, MLPConfig},
    forward_pass::state::ArrayId,
    parameters::ParameterTree,
};

pub fn linear_block<const N: usize>(
    config: &LinearConfig,
    _has_biases: bool,
    input_dimension: usize,
    output_dimensions: [usize; N],
    context: &MTLContext,
    parameter_tree: &ParameterTree<MTLContext>,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
) -> Result<Box<dyn EncodableBlock<Metal>>, MTLError> {
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
            ..
        } => Err(MTLError::Generic("QLoRA linear layer not supported".to_string())),
    }
}

/// Creates an MLP block using the unfused implementation (separate up, activation, down)
pub fn mlp_block(
    config: &MLPConfig,
    model_dimension: usize,
    hidden_dimension: usize,
    context: &MTLContext,
    parameter_tree: &ParameterTree<MTLContext>,
) -> Result<Box<dyn EncodableBlock<Metal>>, MTLError> {
    if let crate::config::MLPConfig::Dense(dense_config) = config {
        let data_type: DataType = dense_config.linear_config.activation_precision().into();

        // Up projection (outputs 2*hidden_dimension for gate and up)
        let up_projection = linear_block(
            &dense_config.linear_config,
            false,
            model_dimension,
            [2 * hidden_dimension],
            context,
            &parameter_tree.subtree("up_projection").map_err(|error| MTLError::Generic(format!("{:?}", error)))?,
            ArrayId::Main,
            ArrayId::MlpFusedUp,
        )?;

        // Gate activation + multiply
        let gate_activation =
            MlpGateActMulEncodable::new(context, data_type, dense_config.activation.clone(), hidden_dimension)?;

        // Down projection
        let down_projection = linear_block(
            &dense_config.linear_config,
            false,
            hidden_dimension,
            [model_dimension],
            context,
            &parameter_tree.subtree("down_projection").map_err(|error| MTLError::Generic(format!("{:?}", error)))?,
            ArrayId::MlpHidden,
            ArrayId::Main,
        )?;

        return Ok(Box::new(MlpBlock::new(up_projection, gate_activation, down_projection)));
    }

    if let crate::config::MLPConfig::MixtureOfExperts(mixture_of_experts_config) = config {
        let mixture_of_experts_block =
            MoeBlock::new(context, mixture_of_experts_config, model_dimension, hidden_dimension, parameter_tree)?;
        return Ok(Box::new(mixture_of_experts_block));
    }

    unreachable!("Unknown MLP config")
}

/// Creates an MLP block using the fused implementation
/// (up projection + activation fused into single kernel)
pub fn mlp_fused_block(
    config: &MLPConfig,
    model_dimension: usize,
    hidden_dimension: usize,
    context: Rc<MTLContext>,
    parameter_tree: &ParameterTree<MTLContext>,
) -> Result<Box<dyn EncodableBlock<Metal>>, MTLError> {
    if let crate::config::MLPConfig::Dense(dense_config) = config {
        match &dense_config.linear_config {
            LinearConfig::Quantized(quantization_config) | LinearConfig::MLXQuantized(quantization_config) => {
                let data_type: DataType = dense_config.linear_config.activation_precision().into();

                // Load up_projection weights directly for fused kernel
                let up_projection_tree = parameter_tree
                    .subtree("up_projection")
                    .map_err(|error| MTLError::Generic(format!("{:?}", error)))?;

                let up_projection_weights = up_projection_tree
                    .leaf("weights")
                    .map_err(|error| MTLError::Generic(format!("Failed to load up weights: {:?}", error)))?;
                let up_projection_weights_buffer = up_projection_weights.buffer().into();

                let up_projection_scales = up_projection_tree
                    .leaf("scales")
                    .map_err(|error| MTLError::Generic(format!("Failed to load up scales: {:?}", error)))?;
                let up_projection_scales_buffer = up_projection_scales.buffer().into();

                // Load zero_points or biases depending on quantization type
                let (up_projection_zero_points_or_biases_buffer, quantization_type) =
                    if let Ok(biases) = up_projection_tree.leaf("biases") {
                        (biases.buffer().into(), QuantizationType::Mlx)
                    } else if let Ok(zero_points) = up_projection_tree.leaf("zero_points") {
                        (zero_points.buffer().into(), QuantizationType::ZeroPoint)
                    } else {
                        return Err(MTLError::Generic(
                            "Missing zero_points or biases for quantized up_projection".to_string(),
                        ));
                    };

                // Create down projection as separate linear
                let down_projection = QuantizedLinear::new(
                    &context,
                    quantization_config,
                    hidden_dimension,
                    model_dimension,
                    &parameter_tree
                        .subtree("down_projection")
                        .map_err(|error| MTLError::Generic(format!("{:?}", error)))?,
                    ArrayId::MlpHidden,
                    ArrayId::Main,
                )?;

                let fused_block = MlpFusedBlock::new_quantized(
                    context,
                    data_type,
                    up_projection_weights_buffer,
                    up_projection_scales_buffer,
                    up_projection_zero_points_or_biases_buffer,
                    model_dimension,
                    hidden_dimension,
                    quantization_config.group_size,
                    quantization_config.weight_quantization_mode,
                    quantization_type,
                    &dense_config.activation,
                    Box::new(down_projection),
                    ArrayId::Main,
                    ArrayId::MlpHidden,
                )?;
                return Ok(Box::new(fused_block));
            },
            LinearConfig::FullPrecision {
                precision,
            } => {
                let data_type: DataType = (*precision).into();

                // Load up_projection weights directly for fused kernel
                let up_projection_tree = parameter_tree
                    .subtree("up_projection")
                    .map_err(|error| MTLError::Generic(format!("{:?}", error)))?;

                let up_projection_weights = up_projection_tree
                    .leaf("weights")
                    .map_err(|error| MTLError::Generic(format!("Failed to load up weights: {:?}", error)))?;
                let up_projection_weights_buffer = up_projection_weights.buffer().into();

                // Create down projection as separate linear
                let down_projection = FullPrecisionLinear::new(
                    &context,
                    data_type,
                    hidden_dimension,
                    model_dimension,
                    &parameter_tree
                        .subtree("down_projection")
                        .map_err(|error| MTLError::Generic(format!("{:?}", error)))?,
                    ArrayId::MlpHidden,
                    ArrayId::Main,
                )?;

                let fused_block = MlpFusedBlock::new_full_precision(
                    context,
                    data_type,
                    up_projection_weights_buffer,
                    model_dimension,
                    hidden_dimension,
                    &dense_config.activation,
                    Box::new(down_projection),
                    ArrayId::Main,
                    ArrayId::MlpHidden,
                )?;
                return Ok(Box::new(fused_block));
            },
            LinearConfig::QLoRA {
                ..
            } => {
                return Err(MTLError::Generic("QLoRA MLP not supported".to_string()));
            },
        }
    }

    if let crate::config::MLPConfig::MixtureOfExperts(mixture_of_experts_config) = config {
        let mixture_of_experts_block =
            MoeBlock::new(&context, mixture_of_experts_config, model_dimension, hidden_dimension, parameter_tree)?;
        return Ok(Box::new(mixture_of_experts_block));
    }

    unreachable!("Unknown MLP config")
}

pub fn embed_block(
    config: &DecoderConfig,
    context: &MTLContext,
    parameter_tree: &ParameterTree<MTLContext>,
) -> Box<dyn EncodableBlock<Metal>> {
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

pub fn readout_block(
    config: &DecoderConfig,
    context: &MTLContext,
    parameter_tree: &ParameterTree<MTLContext>,
) -> Box<dyn EncodableBlock<Metal>> {
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
