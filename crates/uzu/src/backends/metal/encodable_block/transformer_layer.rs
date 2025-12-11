use std::{collections::HashMap, rc::Rc};

use mpsgraph::{
    CompilationDescriptor, Device as MPSDevice, ExecutableExecutionDescriptor,
    Graph,
};
use objc2::rc::{Retained, autoreleasepool};

use super::{
    EncodableBlock, MlpBlock, MoeBlock, QuantizedEmbeddingLookup,
    QuantizedEmbeddingReadout, QuantizedLinear,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext,
        forward_pass::{ArrayId, IOArrays, MPSGraphBlock},
        graph::{
            embeddings_dequantize_weights_subgraph, embeddings_embed_subgraph,
            embeddings_readout_subgraph, linear_subgraph, mlp_subgraph,
            placeholder, shaped_type,
        },
        kernel::mlp::MlpGateActMulEncodable,
    },
    config::{
        DecoderConfig, EmbeddingConfig, LinearConfig, MLPConfig,
        QuantizationConfig,
    },
    parameters::ParameterTree,
};

fn make_execution_descriptor() -> Retained<ExecutableExecutionDescriptor> {
    let execution_descriptor = ExecutableExecutionDescriptor::new();
    execution_descriptor.set_enable_commit_and_continue(true);
    execution_descriptor
}

pub fn linear_block<const N: usize>(
    config: &LinearConfig,
    has_biases: bool,
    input_dim: usize,
    output_dims: [usize; N],
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    compilation_descriptor: &CompilationDescriptor,
) -> Box<dyn EncodableBlock> {
    if let LinearConfig::Quantized(quant_config) = config {
        let out_sum: usize = output_dims.iter().sum();
        return quantized_linear_block_custom(
            quant_config,
            input_dim,
            out_sum,
            context,
            parameter_tree,
            input_array_id,
            output_array_id,
        )
        .unwrap();
    }
    if let LinearConfig::MLXQuantized(quant_config) = config {
        let out_sum: usize = output_dims.iter().sum();
        return quantized_linear_block_custom(
            quant_config,
            input_dim,
            out_sum,
            context,
            parameter_tree,
            input_array_id,
            output_array_id,
        )
        .unwrap();
    }

    autoreleasepool(|_| {
        let graph = Graph::new();

        let input_shape = [-1, input_dim as isize];
        let input_data_type: DataType = config.activation_precision().into();
        let input_placeholder =
            placeholder(&graph, &input_shape, input_data_type);

        let output = linear_subgraph(
            &graph,
            config,
            input_dim,
            output_dims,
            has_biases,
            &input_placeholder,
            parameter_tree,
        )
        .unwrap();

        let retained_shaped_type = shaped_type(&input_shape, input_data_type);
        let feeds =
            HashMap::from([(&*input_placeholder, &*retained_shaped_type)]);

        let executable = graph.compile(
            &MPSDevice::with_device(&context.device),
            &feeds,
            &[&output],
            None,
            Some(&compilation_descriptor),
        );

        let arguments = IOArrays::new(
            vec![input_array_id].into_boxed_slice(),
            vec![output_array_id].into_boxed_slice(),
        );

        Box::new(MPSGraphBlock::new(
            executable,
            make_execution_descriptor(),
            arguments,
        ))
    })
}

pub fn quantized_linear_block_custom(
    config: &QuantizationConfig,
    input_dim: usize,
    output_dim: usize,
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
) -> Result<Box<dyn EncodableBlock>, crate::backends::metal::MTLError> {
    let block = QuantizedLinear::new(
        context,
        config,
        input_dim,
        output_dim,
        parameter_tree,
        input_array_id,
        output_array_id,
    )?;
    Ok(Box::new(block))
}

pub fn mlp_block(
    config: &MLPConfig,
    model_dim: usize,
    hidden_dim: usize,
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    compilation_descriptor: &CompilationDescriptor,
) -> Box<dyn EncodableBlock> {
    if let crate::config::MLPConfig::Dense(dense) = config {
        if let crate::config::LinearConfig::Quantized(ref quant_config)
        | crate::config::LinearConfig::MLXQuantized(ref quant_config) =
            dense.linear_config
        {
            // Quantized MLP path: up (2H) -> act+mul -> down
            let dtype: DataType =
                dense.linear_config.activation_precision().into();

            // Up fused: Main -> MlpFusedUp
            let up = QuantizedLinear::new(
                context,
                quant_config,
                model_dim,
                2 * hidden_dim,
                &parameter_tree.subtree("up_projection").unwrap(),
                ArrayId::Main,
                ArrayId::MlpFusedUp,
            )
            .expect("Failed to build MLP up quantized block");

            // Activation+mul: MlpFusedUp -> MlpHidden
            let gate_op = MlpGateActMulEncodable::new(
                context,
                dtype,
                dense.activation,
                hidden_dim,
            )
            .expect("Failed to build MLP gate activation kernel");

            // Down: MlpHidden -> Main
            let down = QuantizedLinear::new(
                context,
                quant_config,
                hidden_dim,
                model_dim,
                &parameter_tree.subtree("down_projection").unwrap(),
                ArrayId::MlpHidden,
                ArrayId::Main,
            )
            .expect("Failed to build MLP down quantized block");

            let enc = MlpBlock::new(up, gate_op, down);
            return Box::new(enc);
        }
    }

    // STEP 2: Enable MOE block creation (encode will be minimal)
    if let crate::config::MLPConfig::MixtureOfExperts(moe) = config {
        let moe_block =
            MoeBlock::new(context, moe, model_dim, hidden_dim, parameter_tree)
                .expect("Failed to build MoE block");
        return Box::new(moe_block);
    }

    // Non-quantized Dense path is handled via MPSGraph below
    autoreleasepool(|_| {
        let graph = Graph::new();

        let input_shape = [-1, model_dim as isize];
        let input_data_type: DataType = match config {
            crate::config::MLPConfig::Dense(dense) => {
                dense.linear_config.activation_precision().into()
            },
            crate::config::MLPConfig::MixtureOfExperts(moe) => {
                moe.expert_config.linear_config.activation_precision().into()
            },
        };
        let input_placeholder =
            placeholder(&graph, &input_shape, input_data_type);

        let output = match config {
            crate::config::MLPConfig::Dense(dense) => mlp_subgraph(
                &graph,
                &MLPConfig::Dense(dense.clone()),
                model_dim,
                hidden_dim,
                &input_placeholder,
                parameter_tree,
            )
            .unwrap(),
            crate::config::MLPConfig::MixtureOfExperts(_moe) => {
                unreachable!("MoE uses MoeBlock, not MPSGraph")
            },
        };

        let retained_shaped_type = shaped_type(&input_shape, input_data_type);
        let feeds =
            HashMap::from([(&*input_placeholder, &*retained_shaped_type)]);

        let executable = graph.compile(
            &MPSDevice::with_device(&context.device),
            &feeds,
            &[&output],
            None,
            Some(&compilation_descriptor),
        );

        let arguments = IOArrays::new(
            vec![ArrayId::Main].into_boxed_slice(),
            vec![ArrayId::Main].into_boxed_slice(),
        );

        Box::new(MPSGraphBlock::new(
            executable,
            make_execution_descriptor(),
            arguments,
        ))
    })
}

pub fn embed_block(
    config: &DecoderConfig,
    context: &MTLContext,
    compilation_descriptor: &CompilationDescriptor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Box<dyn EncodableBlock> {
    let graph = Graph::new();

    let input_shape = [-1 as isize];
    let input_data_type = DataType::U64;
    let input_placeholder = placeholder(&graph, &input_shape, input_data_type);
    let input_shaped_type = shaped_type(&input_shape, input_data_type);

    match config.embedding_config {
        EmbeddingConfig::Tied {
            precision,
            ..
        } => {
            let weights_shape =
                [config.vocab_size as isize, config.model_dim as isize];
            let weights_data_type: DataType = precision.into();
            let weights_shaped_type =
                shaped_type(&weights_shape, weights_data_type);
            let weights_placeholder =
                placeholder(&graph, &weights_shape, weights_data_type);

            let output = embeddings_embed_subgraph(
                &graph,
                &config.embedding_config,
                &input_placeholder,
                &weights_placeholder,
            )
            .unwrap();

            let feeds = HashMap::from([
                (&*input_placeholder, &*input_shaped_type),
                (&*weights_placeholder, &*weights_shaped_type),
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
                    vec![ArrayId::TokenIds, ArrayId::EmbeddingsInputWeights]
                        .into_boxed_slice(),
                    vec![ArrayId::Main].into_boxed_slice(),
                ),
            );

            Box::new(block)
        },
        EmbeddingConfig::Untied {
            precision,
            ..
        } => {
            let weights_shape =
                [config.vocab_size as isize, config.model_dim as isize];
            let weights_data_type: DataType = precision.into();
            let weights_shaped_type =
                shaped_type(&weights_shape, weights_data_type);
            let weights_placeholder =
                placeholder(&graph, &weights_shape, weights_data_type);

            let output = embeddings_embed_subgraph(
                &graph,
                &config.embedding_config,
                &input_placeholder,
                &weights_placeholder,
            )
            .unwrap();

            let feeds = HashMap::from([
                (&*input_placeholder, &*input_shaped_type),
                (&*weights_placeholder, &*weights_shaped_type),
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
                    vec![ArrayId::TokenIds, ArrayId::EmbeddingsInputWeights]
                        .into_boxed_slice(),
                    vec![ArrayId::Main].into_boxed_slice(),
                ),
            );

            Box::new(block)
        },
        EmbeddingConfig::MLXSemiQuantizedUntied {
            activation_precision,
            ..
        } => {
            let weights_shape =
                [config.vocab_size as isize, config.model_dim as isize];
            let weights_data_type: DataType = activation_precision.into();
            let weights_shaped_type =
                shaped_type(&weights_shape, weights_data_type);
            let weights_placeholder =
                placeholder(&graph, &weights_shape, weights_data_type);

            let output = embeddings_embed_subgraph(
                &graph,
                &config.embedding_config,
                &input_placeholder,
                &weights_placeholder,
            )
            .unwrap();

            let feeds = HashMap::from([
                (&*input_placeholder, &*input_shaped_type),
                (&*weights_placeholder, &*weights_shaped_type),
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
                    vec![ArrayId::TokenIds, ArrayId::EmbeddingsInputWeights]
                        .into_boxed_slice(),
                    vec![ArrayId::Main].into_boxed_slice(),
                ),
            );

            Box::new(block)
        },
        EmbeddingConfig::QuantizedTied {
            embedding_quantization_mode: _,
            ..
        } => {
            // Packed U8 weights with shape [vocab_size, model_dim / 2]
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
            // Use Metal kernel for MLX quantized embeddings
            let data_type: DataType =
                config.output_norm_config.scale_precision.into();

            // Get the embeddings subtree
            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingLookup::new(
                context,
                data_type,
                config.vocab_size,
                config.model_dim,
                group_size,
                embedding_quantization_mode,
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
    let graph = Graph::new();

    let input_shape = [-1 as isize, config.model_dim as isize];
    let input_data_type: DataType =
        config.output_norm_config.scale_precision.into();
    let input_placeholder = placeholder(&graph, &input_shape, input_data_type);
    let input_shaped_type = shaped_type(&input_shape, input_data_type);

    match config.embedding_config {
        EmbeddingConfig::Tied {
            precision,
            ..
        } => {
            let weights_shape =
                [config.vocab_size as isize, config.model_dim as isize];
            let weights_data_type: DataType = precision.into();
            let weights_shaped_type =
                shaped_type(&weights_shape, weights_data_type);
            let weights_placeholder =
                placeholder(&graph, &weights_shape, weights_data_type);

            let weights_transposed =
                graph.transpose(&weights_placeholder, &[1, 0], None);
            let output = embeddings_readout_subgraph(
                &graph,
                &input_placeholder,
                &weights_transposed,
            )
            .unwrap();

            let feeds = HashMap::from([
                (&*input_placeholder, &*input_shaped_type),
                (&*weights_placeholder, &*weights_shaped_type),
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
                    vec![ArrayId::Main, ArrayId::EmbeddingsOutputWeights]
                        .into_boxed_slice(),
                    vec![ArrayId::Logits].into_boxed_slice(),
                ),
            );

            Box::new(block)
        },
        EmbeddingConfig::Untied {
            precision,
            ..
        } => {
            let weights_shape =
                [config.vocab_size as isize, config.model_dim as isize];
            let weights_data_type: DataType = precision.into();
            let weights_shaped_type =
                shaped_type(&weights_shape, weights_data_type);
            let weights_placeholder =
                placeholder(&graph, &weights_shape, weights_data_type);

            let weights_transposed =
                graph.transpose(&weights_placeholder, &[1, 0], None);
            let output = embeddings_readout_subgraph(
                &graph,
                &input_placeholder,
                &weights_transposed,
            )
            .unwrap();

            let feeds = HashMap::from([
                (&*input_placeholder, &*input_shaped_type),
                (&*weights_placeholder, &*weights_shaped_type),
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
                    vec![ArrayId::Main, ArrayId::EmbeddingsOutputWeights]
                        .into_boxed_slice(),
                    vec![ArrayId::Logits].into_boxed_slice(),
                ),
            );

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

            let block = QuantizedEmbeddingReadout::new(
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
            // Packed U8 weights with shape [vocab_size, model_dim / 2]
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
                Some(&compilation_descriptor),
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
            // Use Metal kernel for MLX quantized readout
            let data_type: DataType =
                config.output_norm_config.scale_precision.into();

            // Get the embeddings subtree
            let embeddings_tree = parameter_tree
                .subtree("embedding")
                .expect("Failed to get embedding subtree");

            let block = QuantizedEmbeddingReadout::new(
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
