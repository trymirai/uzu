use std::{collections::HashMap, rc::Rc};

use mpsgraph::{
    CompilationDescriptor, Device as MPSDevice, ExecutableExecutionDescriptor,
    Graph, GraphTensorShapeOps,
};
use objc2::rc::{Retained, autoreleasepool};

use super::{
    super::{
        MTLContext,
        graph::{
            linear_subgraph, mlp_subgraph, placeholder, rms_norm_subgraph,
            shaped_type,
        },
    },
    io_arrays::IOArrays,
    mpsgraph_block::MPSGraphBlock,
    state::ArrayId,
};
use crate::{
    DataType,
    backends::metal::graph::{
        embeddings_dequantize_weights_subgraph, embeddings_embed_subgraph,
        embeddings_readout_subgraph,
    },
    config::{
        DecoderConfig, EmbeddingConfig, LinearConfig, MLPConfig, RMSNormConfig,
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
) -> MPSGraphBlock {
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
            Some(compilation_descriptor),
        );

        let arguments = IOArrays::new(
            vec![input_array_id].into_boxed_slice(),
            vec![output_array_id].into_boxed_slice(),
        );

        MPSGraphBlock::new(executable, make_execution_descriptor(), arguments)
    })
}

pub fn mlp_block(
    config: &MLPConfig,
    model_dim: usize,
    hidden_dim: usize,
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    compilation_descriptor: &CompilationDescriptor,
) -> MPSGraphBlock {
    autoreleasepool(|_| {
        let graph = Graph::new();

        let input_shape = [-1, model_dim as isize];
        let input_data_type: DataType =
            config.linear_config.activation_precision().into();
        let input_placeholder =
            placeholder(&graph, &input_shape, input_data_type);

        let output = mlp_subgraph(
            &graph,
            config,
            model_dim,
            hidden_dim,
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
            Some(compilation_descriptor),
        );

        let arguments = IOArrays::new(
            vec![ArrayId::Main].into_boxed_slice(),
            vec![ArrayId::Main].into_boxed_slice(),
        );

        MPSGraphBlock::new(executable, make_execution_descriptor(), arguments)
    })
}

pub fn rms_norm_block(
    config: &RMSNormConfig,
    model_dim: usize,
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    compilation_descriptor: &CompilationDescriptor,
) -> MPSGraphBlock {
    autoreleasepool(|_| {
        let graph = Graph::new();

        let input_shape = [-1, model_dim as isize];
        let input_data_type: DataType = config.scale_precision.into();
        let input_placeholder =
            placeholder(&graph, &input_shape, input_data_type);

        let output = rms_norm_subgraph(
            &graph,
            config,
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
            Some(compilation_descriptor),
        );

        let arguments = IOArrays::new(
            vec![input_array_id].into_boxed_slice(),
            vec![output_array_id].into_boxed_slice(),
        );

        MPSGraphBlock::new(executable, make_execution_descriptor(), arguments)
    })
}

pub fn embed_block(
    config: &DecoderConfig,
    context: &MTLContext,
    compilation_descriptor: &CompilationDescriptor,
) -> MPSGraphBlock {
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
                Some(compilation_descriptor),
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

            block
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
                Some(compilation_descriptor),
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

            block
        },
        EmbeddingConfig::QuantizedTied {
            embedding_quantization_mode,
            ..
        } => {
            let weights_shape =
                [config.vocab_size as isize, config.model_dim as isize];
            let weights_data_type: DataType =
                embedding_quantization_mode.into();
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
                Some(compilation_descriptor),
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

            block
        },
    }
}

pub fn readout_block(
    config: &DecoderConfig,
    context: &MTLContext,
    compilation_descriptor: &CompilationDescriptor,
) -> MPSGraphBlock {
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
                Some(compilation_descriptor),
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

            block
        },
        EmbeddingConfig::Untied {
            precision,
            ..
        } => {
            let weights_shape =
                [config.model_dim as isize, config.vocab_size as isize];
            let weights_data_type: DataType = precision.into();
            let weights_shaped_type =
                shaped_type(&weights_shape, weights_data_type);
            let weights_placeholder =
                placeholder(&graph, &weights_shape, weights_data_type);

            let output = embeddings_readout_subgraph(
                &graph,
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
                Some(compilation_descriptor),
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

            block
        },
        EmbeddingConfig::QuantizedTied {
            embedding_quantization_mode,
            ..
        } => {
            let weights_shape =
                [config.vocab_size as isize, config.model_dim as isize];
            let weights_data_type: DataType =
                embedding_quantization_mode.into();
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

            block
        },
    }
}
