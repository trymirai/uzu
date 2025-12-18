use std::{collections::HashMap, rc::Rc};

use mpsgraph::{
    CompilationDescriptor, Device as MPSDevice, ExecutableExecutionDescriptor,
    Graph,
};
use objc2::rc::Retained;

use super::{
    EncodableBlock, FullPrecisionEmbeddingLookup,
    FullPrecisionEmbeddingReadout, FullPrecisionLinear, MlpBlock, MoeBlock,
    QuantizedEmbeddingLookup, QuantizedEmbeddingReadout, QuantizedLinear,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, IOArrays, MPSGraphBlock},
        graph::{
            embeddings_dequantize_weights_subgraph, embeddings_embed_subgraph,
            embeddings_readout_subgraph, placeholder, shaped_type,
        },
        kernel::mlp::MlpGateActMulEncodable,
    },
    config::{DecoderConfig, EmbeddingConfig, LinearConfig, MLPConfig},
    parameters::ParameterTree,
};

fn make_execution_descriptor() -> Retained<ExecutableExecutionDescriptor> {
    let execution_descriptor = ExecutableExecutionDescriptor::new();
    execution_descriptor.set_enable_commit_and_continue(true);
    execution_descriptor
}

pub fn linear_block<const N: usize>(
    config: &LinearConfig,
    _has_biases: bool,
    input_dim: usize,
    output_dims: [usize; N],
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    _compilation_descriptor: &CompilationDescriptor,
) -> Result<Box<dyn EncodableBlock>, MTLError> {
    let out_sum: usize = output_dims.iter().sum();
    match config {
        LinearConfig::Quantized(quant_config)
        | LinearConfig::MLXQuantized(quant_config) => {
            let block = QuantizedLinear::new(
                context,
                quant_config,
                input_dim,
                out_sum,
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
                input_dim,
                out_sum,
                parameter_tree,
                input_array_id,
                output_array_id,
            )?;
            Ok(Box::new(block))
        },
        LinearConfig::QLoRA {
            ..
        } => Err(MTLError::Generic(
            "QLoRA linear layer not supported".to_string(),
        )),
    }
}

pub fn mlp_block(
    config: &MLPConfig,
    model_dim: usize,
    hidden_dim: usize,
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    _compilation_descriptor: &CompilationDescriptor,
) -> Result<Box<dyn EncodableBlock>, MTLError> {
    if let crate::config::MLPConfig::Dense(dense) = config {
        match &dense.linear_config {
            LinearConfig::Quantized(quant_config)
            | LinearConfig::MLXQuantized(quant_config) => {
                let dtype: DataType =
                    dense.linear_config.activation_precision().into();

                let up = QuantizedLinear::new(
                    context,
                    quant_config,
                    model_dim,
                    2 * hidden_dim,
                    &parameter_tree
                        .subtree("up_projection")
                        .map_err(|e| MTLError::Generic(format!("{:?}", e)))?,
                    ArrayId::Main,
                    ArrayId::MlpFusedUp,
                )?;

                let gate_op = MlpGateActMulEncodable::new(
                    context,
                    dtype,
                    dense.activation,
                    hidden_dim,
                )?;

                let down = QuantizedLinear::new(
                    context,
                    quant_config,
                    hidden_dim,
                    model_dim,
                    &parameter_tree
                        .subtree("down_projection")
                        .map_err(|e| MTLError::Generic(format!("{:?}", e)))?,
                    ArrayId::MlpHidden,
                    ArrayId::Main,
                )?;

                let enc = MlpBlock::new(Box::new(up), gate_op, Box::new(down));
                return Ok(Box::new(enc));
            },
            LinearConfig::FullPrecision {
                precision,
            } => {
                let dtype: DataType = (*precision).into();

                let up = FullPrecisionLinear::new(
                    context,
                    dtype,
                    model_dim,
                    2 * hidden_dim,
                    &parameter_tree
                        .subtree("up_projection")
                        .map_err(|e| MTLError::Generic(format!("{:?}", e)))?,
                    ArrayId::Main,
                    ArrayId::MlpFusedUp,
                )?;

                let gate_op = MlpGateActMulEncodable::new(
                    context,
                    dtype,
                    dense.activation,
                    hidden_dim,
                )?;

                let down = FullPrecisionLinear::new(
                    context,
                    dtype,
                    hidden_dim,
                    model_dim,
                    &parameter_tree
                        .subtree("down_projection")
                        .map_err(|e| MTLError::Generic(format!("{:?}", e)))?,
                    ArrayId::MlpHidden,
                    ArrayId::Main,
                )?;

                let enc = MlpBlock::new(Box::new(up), gate_op, Box::new(down));
                return Ok(Box::new(enc));
            },
            LinearConfig::QLoRA {
                ..
            } => {
                return Err(MTLError::Generic(
                    "QLoRA MLP not supported".to_string(),
                ));
            },
        }
    }

    if let crate::config::MLPConfig::MixtureOfExperts(moe) = config {
        let moe_block =
            MoeBlock::new(context, moe, model_dim, hidden_dim, parameter_tree)?;
        return Ok(Box::new(moe_block));
    }

    unreachable!("Unknown MLP config")
}

pub fn embed_block(
    config: &DecoderConfig,
    context: &MTLContext,
    compilation_descriptor: &CompilationDescriptor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Box<dyn EncodableBlock> {
    match &config.embedding_config {
        EmbeddingConfig::Tied {
            common,
            precision,
        } => {
            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

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
            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

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
            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

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

            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingLookup::new_untied_input(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                *group_size,
                *embedding_quantization_mode,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding lookup kernel");

            Box::new(block)
        },
        EmbeddingConfig::QuantizedTied {
            embedding_quantization_mode: _,
            ..
        } => {
            let graph = Graph::new();

            let input_shape = [-1 as isize];
            let input_data_type = DataType::U64;
            let input_placeholder =
                placeholder(&graph, &input_shape, input_data_type);
            let input_shaped_type = shaped_type(&input_shape, input_data_type);
            let weights_shape =
                [config.vocab_size as isize, (config.model_dim / 2) as isize];
            let weights_data_type = DataType::U8;
            let weights_shaped_type =
                shaped_type(&weights_shape, weights_data_type);
            let weights_placeholder =
                placeholder(&graph, &weights_shape, weights_data_type);

            let scales_shape = [config.vocab_size as isize];
            let scales_data_type: DataType =
                config.output_norm_config.scale_precision.into();
            let scales_shaped_type =
                shaped_type(&scales_shape, scales_data_type);
            let scales_placeholder =
                placeholder(&graph, &scales_shape, scales_data_type);

            let weights = embeddings_dequantize_weights_subgraph(
                &graph,
                &weights_placeholder,
                &scales_placeholder,
            )
            .unwrap();

            let output = embeddings_embed_subgraph(
                &graph,
                &config.embedding_config,
                &input_placeholder,
                &weights,
            )
            .unwrap();

            let feeds = HashMap::from([
                (&*input_placeholder, &*input_shaped_type),
                (&*weights_placeholder, &*weights_shaped_type),
                (&*scales_placeholder, &*scales_shaped_type),
            ]);

            let executable = graph.compile(
                &MPSDevice::with_device(&context.device),
                &feeds,
                &[&output],
                None,
                Some(&compilation_descriptor),
            );

            let block = MPSGraphBlock::new(
                executable,
                make_execution_descriptor(),
                IOArrays::new(
                    vec![
                        ArrayId::TokenIds,
                        ArrayId::EmbeddingsInputWeights,
                        ArrayId::EmbeddingsScales,
                    ]
                    .into_boxed_slice(),
                    vec![ArrayId::Main].into_boxed_slice(),
                ),
            );

            Box::new(block)
        },
        EmbeddingConfig::MLXQuantizedTied {
            group_size,
            embedding_quantization_mode,
            ..
        } => {
            let data_type: DataType =
                config.output_norm_config.scale_precision.into();
            let input_scale =
                config.embedding_config.common().input_scale.unwrap_or(1.0);

            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

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
        EmbeddingConfig::MLXQuantizedUntied {
            group_size,
            embedding_quantization_mode,
            ..
        } => {
            let data_type: DataType =
                config.output_norm_config.scale_precision.into();
            let input_scale =
                config.embedding_config.common().input_scale.unwrap_or(1.0);

            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

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
    }
}

pub fn readout_block(
    config: &DecoderConfig,
    context: &MTLContext,
    compilation_descriptor: &CompilationDescriptor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Box<dyn EncodableBlock> {
    match config.embedding_config {
        EmbeddingConfig::Tied {
            precision,
            ..
        } => {
            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

            let block = FullPrecisionEmbeddingReadout::new(
                context,
                precision.into(),
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
            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

            let block = FullPrecisionEmbeddingReadout::new(
                context,
                precision.into(),
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
            let data_type: DataType = activation_precision.into();
            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingReadout::new_untied_output(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                group_size,
                embedding_quantization_mode,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding readout kernel");

            Box::new(block)
        },
        EmbeddingConfig::QuantizedTied {
            embedding_quantization_mode: _,
            ..
        } => {
            let graph = Graph::new();

            let input_shape = [-1 as isize, config.model_dim as isize];
            let input_data_type: DataType =
                config.output_norm_config.scale_precision.into();
            let input_placeholder =
                placeholder(&graph, &input_shape, input_data_type);
            let input_shaped_type = shaped_type(&input_shape, input_data_type);

            let weights_shape =
                [config.vocab_size as isize, (config.model_dim / 2) as isize];
            let weights_data_type = DataType::U8;
            let weights_shaped_type =
                shaped_type(&weights_shape, weights_data_type);
            let weights_placeholder =
                placeholder(&graph, &weights_shape, weights_data_type);

            let scales_shape = [config.vocab_size as isize];
            let scales_data_type: DataType =
                config.output_norm_config.scale_precision.into();
            let scales_shaped_type =
                shaped_type(&scales_shape, scales_data_type);
            let scales_placeholder =
                placeholder(&graph, &scales_shape, scales_data_type);

            let weights = embeddings_dequantize_weights_subgraph(
                &graph,
                &weights_placeholder,
                &scales_placeholder,
            )
            .unwrap();
            let weights_transposed = graph.transpose(&weights, &[1, 0], None);
            let output = embeddings_readout_subgraph(
                &graph,
                &input_placeholder,
                &weights_transposed,
            )
            .unwrap();

            let feeds = HashMap::from([
                (&*input_placeholder, &*input_shaped_type),
                (&*weights_placeholder, &*weights_shaped_type),
                (&*scales_placeholder, &*scales_shaped_type),
            ]);

            let executable = graph.compile(
                &MPSDevice::with_device(&context.device),
                &feeds,
                &[&output],
                None,
                Some(compilation_descriptor),
            );

            let block = MPSGraphBlock::new(
                executable,
                make_execution_descriptor(),
                IOArrays::new(
                    vec![
                        ArrayId::Main,
                        ArrayId::EmbeddingsOutputWeights,
                        ArrayId::EmbeddingsScales,
                    ]
                    .into_boxed_slice(),
                    vec![ArrayId::Logits].into_boxed_slice(),
                ),
            );

            Box::new(block)
        },
        EmbeddingConfig::MLXQuantizedTied {
            group_size,
            embedding_quantization_mode,
            ..
        } => {
            let data_type: DataType =
                config.output_norm_config.scale_precision.into();

            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingReadout::new_tied(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                group_size,
                embedding_quantization_mode,
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
            let data_type: DataType =
                config.output_norm_config.scale_precision.into();

            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingReadout::new_untied_output(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                group_size,
                embedding_quantization_mode,
                &embeddings_tree,
            )
            .expect("Failed to create quantized embedding readout kernel");

            Box::new(block)
        },
    }
}
