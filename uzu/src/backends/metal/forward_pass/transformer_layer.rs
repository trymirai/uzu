use std::{collections::HashMap, rc::Rc};

use mpsgraph::{
    CompilationDescriptor, Device as MPSDevice, Executable,
    ExecutableExecutionDescriptor, Graph, GraphTensorShapeOps,
};
use objc2::rc::{Retained, autoreleasepool};

use super::{
    super::{
        MTLContext,
        graph::{
            attention_subgraph, linear_subgraph, mlp_subgraph, placeholder,
            rms_norm_subgraph, shaped_type,
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
        embeddings_readout_subgraph, rotation_subgraph,
    },
    config::{
        DecoderConfig, EmbeddingConfig, LinearConfig, MLPConfig, RMSNormConfig,
        RoPEConfig,
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

pub fn rotation_executable(
    rope_name: String,
    rope_config: &RoPEConfig,
    head_dim: usize,
    num_groups: usize,
    num_heads: usize,
    context_length: usize,
    context: &MTLContext,
    root_parameter_tree: &ParameterTree<Rc<MTLContext>>,
    compilation_descriptor: &Retained<CompilationDescriptor>,
) -> Retained<Executable> {
    autoreleasepool(|_| {
        let graph = Graph::new();

        let activation_data_type: DataType =
            rope_config.common().precision.into();

        let qkv_shape =
            [-1, ((num_heads + 2 * num_groups) * head_dim) as isize];
        let qkv_placeholder =
            placeholder(&graph, &qkv_shape, activation_data_type);
        let qkv_shaped_type = shaped_type(&qkv_shape, activation_data_type);

        let token_positions_placeholder =
            placeholder(&graph, &[-1 as isize], DataType::I32);
        let token_positions_shaped_type =
            shaped_type(&[-1 as isize], DataType::I32);

        let (output_queries, output_keys) = rotation_subgraph(
            &graph,
            rope_name,
            rope_config,
            head_dim,
            num_groups,
            num_heads,
            context_length,
            &qkv_placeholder,
            &token_positions_placeholder,
            root_parameter_tree,
        )
        .unwrap();

        let feeds = HashMap::from([
            (&*qkv_placeholder, &*qkv_shaped_type),
            (&*token_positions_placeholder, &*token_positions_shaped_type),
        ]);

        graph.compile(
            &MPSDevice::with_device(&context.device),
            &feeds,
            &[&output_queries, &output_keys],
            Some(compilation_descriptor),
        )
    })
}

pub fn attention_executable(
    activation_data_type: DataType,
    head_dim: usize,
    num_groups: usize,
    num_heads: usize,
    attention_scale: Option<f32>,
    context: &MTLContext,
    suffix_length: usize,
    prefix_length: usize,
    compilation_descriptor: &CompilationDescriptor,
) -> Retained<Executable> {
    autoreleasepool(|_| {
        let graph = Graph::new();

        let qkv_shape =
            [-1, ((num_heads + 2 * num_groups) * head_dim) as isize];
        let qkv_placeholder =
            placeholder(&graph, &qkv_shape, activation_data_type);
        let retained_qkv_shaped_type =
            shaped_type(&qkv_shape, activation_data_type);

        let kv_cache_shape = [num_groups as isize, -1, head_dim as isize];
        let key_cache_placeholder =
            placeholder(&graph, &kv_cache_shape, activation_data_type);
        let value_cache_placeholder =
            placeholder(&graph, &kv_cache_shape, activation_data_type);
        let retained_kv_cache_shaped_type =
            shaped_type(&kv_cache_shape, activation_data_type);

        let rotated_queries_shape = [num_heads as isize, -1, head_dim as isize];
        let rotated_queries_placeholder =
            placeholder(&graph, &rotated_queries_shape, activation_data_type);
        let rotated_queries_shaped_type =
            shaped_type(&rotated_queries_shape, activation_data_type);

        let rotated_keys_shape = [num_groups as isize, -1, head_dim as isize];
        let rotated_keys_placeholder =
            placeholder(&graph, &rotated_keys_shape, activation_data_type);
        let rotated_keys_shaped_type =
            shaped_type(&rotated_keys_shape, activation_data_type);

        let bias_placeholder =
            placeholder(&graph, &[-1, -1], activation_data_type);
        let retained_bias_shaped_type =
            shaped_type(&[-1, -1], activation_data_type);

        let output = attention_subgraph(
            &graph,
            head_dim,
            num_groups,
            num_heads,
            attention_scale,
            &qkv_placeholder,
            &key_cache_placeholder,
            &value_cache_placeholder,
            &rotated_queries_placeholder,
            &rotated_keys_placeholder,
            &bias_placeholder,
            suffix_length,
            prefix_length,
        )
        .unwrap();

        let feeds = HashMap::from([
            (&*qkv_placeholder, &*retained_qkv_shaped_type),
            (&*key_cache_placeholder, &*retained_kv_cache_shaped_type),
            (&*value_cache_placeholder, &*retained_kv_cache_shaped_type),
            (&*rotated_queries_placeholder, &*rotated_queries_shaped_type),
            (&*rotated_keys_placeholder, &*rotated_keys_shaped_type),
            (&bias_placeholder, &retained_bias_shaped_type),
        ]);

        graph.compile(
            &MPSDevice::with_device(&context.device),
            &feeds,
            &[&output],
            Some(compilation_descriptor),
        )
    })
}

pub fn rotation_block(executable: Retained<Executable>) -> MPSGraphBlock {
    autoreleasepool(|_| {
        let input_array_ids = vec![ArrayId::QKV, ArrayId::TokenPositions];
        let output_array_ids =
            vec![ArrayId::RotatedQueries, ArrayId::RotatedKeys];

        let arguments = IOArrays::new(
            input_array_ids.into_boxed_slice(),
            output_array_ids.into_boxed_slice(),
        );

        MPSGraphBlock::new(executable, make_execution_descriptor(), arguments)
    })
}

pub fn attention_block(
    executable: Retained<Executable>,
    layer_index: usize,
) -> MPSGraphBlock {
    autoreleasepool(|_| {
        let input_array_ids = vec![
            ArrayId::QKV,
            ArrayId::Keys(layer_index),
            ArrayId::Values(layer_index),
            ArrayId::RotatedQueries,
            ArrayId::RotatedKeys,
        ];
        let output_array_ids = vec![ArrayId::AttentionOutput];

        let arguments = IOArrays::new(
            input_array_ids.into_boxed_slice(),
            output_array_ids.into_boxed_slice(),
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
