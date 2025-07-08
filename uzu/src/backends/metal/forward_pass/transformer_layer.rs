use std::{collections::HashMap, rc::Rc};

use mpsgraph::{
    CompilationDescriptor, Device as MPSDevice, Executable,
    ExecutableExecutionDescriptor, Graph, ShapedType, Tensor,
};
use objc2::rc::{Retained, autoreleasepool};

use super::{
    super::{
        MTLContext,
        graph::{
            EmbeddingParams, attention_subgraph, embed_callable_subgraph,
            embed_placeholder_weights_subgraph, embedding_params,
            linear_subgraph, mlp_subgraph, placeholder,
            readout_callable_subgraph, readout_placeholder_weights_subgraph,
            readout_subgraph, rms_norm_subgraph, shaped_type,
        },
    },
    io_arrays::IOArrays,
    mpsgraph_block::MPSGraphBlock,
    state::ArrayId,
};
use crate::{
    DataType,
    backends::metal::graph::rotation_subgraph,
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

pub struct EmbeddingBlocks {
    pub embedding: Option<MPSGraphBlock>,
    pub embed: MPSGraphBlock,
    pub readout: MPSGraphBlock,
}

pub fn embedding_callable_blocks(
    config: &DecoderConfig,
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    compilation_descriptor: &CompilationDescriptor,
) -> EmbeddingBlocks {
    autoreleasepool(|_| {
        let embedding_graph = Graph::new();
        let embedding_params = embedding_params(
            &embedding_graph,
            &config.embedding_config,
            config.vocab_size,
            config.model_dim,
            parameter_tree,
        )
        .unwrap();

        match embedding_params {
            EmbeddingParams::Tied(embeddings) => {
                let embedding_callable_name: &str = "embedding";
                let embedding_output = embeddings;
                let embedding_executable = embedding_graph.compile(
                    &MPSDevice::with_device(&context.device),
                    &HashMap::<&Tensor, &ShapedType>::new(),
                    &[&embedding_output],
                    Some(compilation_descriptor),
                );

                let compilation_desctiptor_with_embedding_callable =
                    CompilationDescriptor::new();
                compilation_desctiptor_with_embedding_callable.add_callable(
                    embedding_callable_name,
                    &embedding_executable,
                );

                let embed_graph = Graph::new();
                let embed_input_shape = [-1 as isize];
                let embed_input_data_type = DataType::U64;
                let embed_input_placeholder = placeholder(
                    &embed_graph,
                    &embed_input_shape,
                    embed_input_data_type,
                );
                let retained_embed_input_shaped_type =
                    shaped_type(&embed_input_shape, embed_input_data_type);
                let embed_output = embed_callable_subgraph(
                    &embed_graph,
                    &config.embedding_config,
                    embedding_callable_name,
                    [config.vocab_size as isize, config.model_dim as isize],
                    &embed_input_placeholder,
                )
                .unwrap();

                let embed_feeds = HashMap::from([(
                    &*embed_input_placeholder,
                    &*retained_embed_input_shaped_type,
                )]);

                let embed_executable = embed_graph.compile(
                    &MPSDevice::with_device(&context.device),
                    &embed_feeds,
                    &[&embed_output],
                    Some(&compilation_desctiptor_with_embedding_callable),
                );

                let readout_graph = Graph::new();
                let readout_input_shape =
                    [-1 as isize, config.model_dim as isize];
                let readout_input = placeholder(
                    &readout_graph,
                    &readout_input_shape,
                    config.output_norm_config.scale_precision.into(),
                );
                let retained_readout_input_shaped_type = shaped_type(
                    &readout_input_shape,
                    config.output_norm_config.scale_precision.into(),
                );
                let readout_output = readout_callable_subgraph(
                    &readout_graph,
                    &config.embedding_config,
                    embedding_callable_name,
                    [config.vocab_size as isize, config.model_dim as isize],
                    &readout_input,
                )
                .unwrap();

                let readout_feeds = HashMap::from([(
                    &*readout_input,
                    &*retained_readout_input_shaped_type,
                )]);

                let readout_executable = readout_graph.compile(
                    &MPSDevice::with_device(&context.device),
                    &readout_feeds,
                    &[&readout_output],
                    Some(&compilation_desctiptor_with_embedding_callable),
                );

                let embedding_block = MPSGraphBlock::new(
                    embedding_executable,
                    make_execution_descriptor(),
                    IOArrays::new(Box::new([]), Box::new([])),
                );
                let embed_block = MPSGraphBlock::new(
                    embed_executable,
                    make_execution_descriptor(),
                    IOArrays::new(
                        vec![ArrayId::TokenIds].into_boxed_slice(),
                        vec![ArrayId::Main].into_boxed_slice(),
                    ),
                );
                let readout_block = MPSGraphBlock::new(
                    readout_executable,
                    make_execution_descriptor(),
                    IOArrays::new(
                        Box::new([ArrayId::Main]),
                        Box::new([ArrayId::Logits]),
                    ),
                );
                EmbeddingBlocks {
                    embedding: Some(embedding_block),
                    embed: embed_block,
                    readout: readout_block,
                }
            },
            EmbeddingParams::Untied {
                input_weights: _,
                output_weights: _,
            } => {
                unimplemented!()
            },
        }
    })
}

pub fn embedding_blocks(
    config: &DecoderConfig,
    context: &MTLContext,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    compilation_descriptor: &CompilationDescriptor,
) -> EmbeddingBlocks {
    autoreleasepool(|_| {
        let embed_graph = Graph::new();
        let embed_input_shape = [-1 as isize];
        let embed_input_data_type = DataType::U64;
        let embed_input_placeholder = placeholder(
            &embed_graph,
            &embed_input_shape,
            embed_input_data_type,
        );
        let retained_embed_input_shaped_type =
            shaped_type(&embed_input_shape, embed_input_data_type);
        let embed_output = embed_subgraph(
            &embed_graph,
            &config.embedding_config,
            config.vocab_size,
            config.model_dim,
            &embed_input_placeholder,
            parameter_tree,
        )
        .unwrap();

        let embed_feeds = HashMap::from([(
            &*embed_input_placeholder,
            &*retained_embed_input_shaped_type,
        )]);

        let embed_executable = embed_graph.compile(
            &MPSDevice::with_device(&context.device),
            &embed_feeds,
            &[&embed_output],
            Some(compilation_descriptor),
        );

        let readout_graph = Graph::new();
        let readout_input_shape = [-1 as isize, config.model_dim as isize];
        let readout_input = placeholder(
            &readout_graph,
            &readout_input_shape,
            config.output_norm_config.scale_precision.into(),
        );
        let retained_readout_input_shaped_type = shaped_type(
            &readout_input_shape,
            config.output_norm_config.scale_precision.into(),
        );
        let readout_output = readout_subgraph(
            &readout_graph,
            &config.embedding_config,
            config.vocab_size,
            config.model_dim,
            &readout_input,
            parameter_tree,
        )
        .unwrap();

        let readout_feeds = HashMap::from([(
            &*readout_input,
            &*retained_readout_input_shaped_type,
        )]);

        let readout_executable = readout_graph.compile(
            &MPSDevice::with_device(&context.device),
            &readout_feeds,
            &[&readout_output],
            Some(compilation_descriptor),
        );

        let embed_block = MPSGraphBlock::new(
            embed_executable,
            make_execution_descriptor(),
            IOArrays::new(
                vec![ArrayId::TokenIds].into_boxed_slice(),
                vec![ArrayId::Main].into_boxed_slice(),
            ),
        );
        let readout_block = MPSGraphBlock::new(
            readout_executable,
            make_execution_descriptor(),
            IOArrays::new(
                Box::new([ArrayId::Main]),
                Box::new([ArrayId::Logits]),
            ),
        );
        EmbeddingBlocks {
            embedding: None,
            embed: embed_block,
            readout: readout_block,
        }
    })
}

pub fn embedding_with_placeholder_weights_blocks(
    config: &DecoderConfig,
    context: &MTLContext,
    compilation_descriptor: &CompilationDescriptor,
) -> EmbeddingBlocks {
    autoreleasepool(|_| {
        let embed_weights_shape =
            [config.vocab_size as isize, config.model_dim as isize];
        let embed_weights_data_type: DataType =
            config.output_norm_config.scale_precision.into();
        let embed_weights_shaped_type =
            shaped_type(&embed_weights_shape, embed_weights_data_type);

        let embed_graph = Graph::new();
        let embed_input_shape = [-1 as isize];
        let embed_input_data_type = DataType::U64;
        let embed_input_placeholder = placeholder(
            &embed_graph,
            &embed_input_shape,
            embed_input_data_type,
        );
        let embed_input_shaped_type =
            shaped_type(&embed_input_shape, embed_input_data_type);
        let embed_weights_placeholder = placeholder(
            &embed_graph,
            &embed_weights_shape,
            embed_weights_data_type,
        );
        let embed_output = embed_placeholder_weights_subgraph(
            &embed_graph,
            &config.embedding_config,
            &embed_input_placeholder,
            &embed_weights_placeholder,
        )
        .unwrap();

        let embed_feeds = HashMap::from([
            (&*embed_input_placeholder, &*embed_input_shaped_type),
            (&*embed_weights_placeholder, &*embed_weights_shaped_type),
        ]);

        let embed_executable = embed_graph.compile(
            &MPSDevice::with_device(&context.device),
            &embed_feeds,
            &[&embed_output],
            Some(compilation_descriptor),
        );

        let embed_block = MPSGraphBlock::new(
            embed_executable,
            make_execution_descriptor(),
            IOArrays::new(
                vec![ArrayId::TokenIds, ArrayId::EmbeddingsInput]
                    .into_boxed_slice(),
                vec![ArrayId::Main].into_boxed_slice(),
            ),
        );

        let readout_weights_shape: [isize; 2];
        let readout_weights_data_type: DataType;
        let readout_transpose_weights: bool;
        match config.embedding_config {
            EmbeddingConfig::Tied {
                common: _,
                precision,
            } => {
                readout_weights_shape =
                    [config.vocab_size as isize, config.model_dim as isize];
                readout_weights_data_type = precision.into();
                readout_transpose_weights = true;
            },
            EmbeddingConfig::Untied {
                common: _,
                precision,
            } => {
                readout_weights_shape =
                    [config.model_dim as isize, config.vocab_size as isize];
                readout_weights_data_type = precision.into();
                readout_transpose_weights = false;
            },
            _ => {
                unimplemented!()
            },
        }
        let readout_weights_shaped_type =
            shaped_type(&readout_weights_shape, readout_weights_data_type);

        let readout_graph = Graph::new();
        let readout_input_shape = [-1 as isize, config.model_dim as isize];
        let readout_input = placeholder(
            &readout_graph,
            &readout_input_shape,
            config.output_norm_config.scale_precision.into(),
        );
        let readout_input_shaped_type = shaped_type(
            &readout_input_shape,
            config.output_norm_config.scale_precision.into(),
        );
        let readout_weights_placeholder = placeholder(
            &readout_graph,
            &readout_weights_shape,
            readout_weights_data_type,
        );
        let readout_output = readout_placeholder_weights_subgraph(
            &readout_graph,
            &readout_input,
            &readout_weights_placeholder,
            readout_transpose_weights,
        )
        .unwrap();

        let readout_feeds = HashMap::from([
            (&*readout_input, &*readout_input_shaped_type),
            (&*readout_weights_placeholder, &*readout_weights_shaped_type),
        ]);

        let readout_executable = readout_graph.compile(
            &MPSDevice::with_device(&context.device),
            &readout_feeds,
            &[&readout_output],
            Some(compilation_descriptor),
        );

        let readout_block = MPSGraphBlock::new(
            readout_executable,
            make_execution_descriptor(),
            IOArrays::new(
                Box::new([ArrayId::Main, ArrayId::EmbeddingsOutput]),
                Box::new([ArrayId::Logits]),
            ),
        );
        EmbeddingBlocks {
            embedding: None,
            embed: embed_block,
            readout: readout_block,
        }
    })
}
