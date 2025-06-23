use std::rc::Rc;

use mpsgraph::{
    Graph, GraphControlFlowOps, GraphLinearAlgebraOps, GraphMemoryOps,
    GraphTensorShapeOps, Tensor,
};
use objc2::{Message, rc::Retained};

use super::{
    common::GraphConstructionError,
    rope::{apply_rope_subgraph, positional_embeddings_subgraph},
};
use crate::{
    backends::metal::{MTLContext, utils::mps_shape},
    config::RoPEConfig,
    parameters::ParameterTree,
};

fn expand_keys_values(
    graph: &Graph,
    keys_or_values: &Tensor,
    num_groups: usize,
    head_dim: usize,
) -> Retained<Tensor> {
    graph.reshape(
        keys_or_values,
        &mps_shape(&[1, (num_groups) as i64, -1_i64, head_dim as i64]).into(),
        None,
    )
}

pub fn rotation_subgraph(
    graph: &Graph,
    rope_name: String,
    rope_config: &RoPEConfig,
    head_dim: usize,
    num_groups: usize,
    num_heads: usize,
    context_length: usize,
    qkv: &Tensor, // [suffix_length, (num_heads + 2 * num_groups) * head_dim]
    token_positions: &Tensor, // u64 [suffix_length]
    root_parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<(Retained<Tensor>, Retained<Tensor>), GraphConstructionError> {
    let positional_embeddings = positional_embeddings_subgraph(
        graph,
        rope_config,
        head_dim,
        context_length,
        token_positions,
        &root_parameter_tree.subtree(rope_name.as_str())?,
    )?;

    let total_query_dim = num_heads * head_dim;
    let total_key_value_dim = num_groups * head_dim;

    let query_slice = graph.slice(qkv, 1, 0, total_query_dim as i64, None);
    let new_key_slice = graph.slice(
        qkv,
        1,
        total_query_dim as i64,
        total_key_value_dim as i64,
        None,
    );

    let queries = graph.reshape(
        &query_slice,
        &mps_shape(&[-1, num_heads as isize, head_dim as isize]),
        None,
    );
    let new_keys = graph.reshape(
        &new_key_slice,
        &mps_shape(&[-1, num_groups as isize, head_dim as isize]),
        None,
    );

    let rotated_queries =
        apply_rope_subgraph(graph, &positional_embeddings, &*queries);
    let rotated_new_keys =
        apply_rope_subgraph(graph, &positional_embeddings, &*new_keys);

    let permuted_queries = graph.transpose(&rotated_queries, &[1, 0, 2], None);
    let permuted_keys = graph.transpose(&rotated_new_keys, &[1, 0, 2], None);

    Ok((permuted_queries, permuted_keys))
}

pub fn attention_subgraph(
    graph: &Retained<Graph>,
    head_dim: usize,
    num_groups: usize,
    num_heads: usize,
    attention_scale: Option<f32>,
    qkv: &Retained<Tensor>, // [suffix_length, (num_heads + 2 * num_groups) * head_dim]
    key_cache: &Retained<Tensor>, // [max_sequence_length, num_groups * head_dim]
    value_cache: &Retained<Tensor>, // [max_sequence_length, num_groups * head_dim]
    permuted_queries: &Retained<Tensor>,
    permuted_keys: &Retained<Tensor>,
    bias: &Retained<Tensor>, // [suffix_length, prefix_length]
    suffix_length: usize,
    prefix_length: usize,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let total_query_dim = num_heads * head_dim;
    let total_key_value_dim = num_groups * head_dim;

    let new_value_slice = graph.slice(
        qkv,
        1,
        (total_query_dim + total_key_value_dim) as i64,
        total_key_value_dim as i64,
        None,
    );

    let new_values = graph.reshape(
        &new_value_slice,
        &mps_shape(&[-1, num_groups as isize, head_dim as isize]),
        None,
    );

    let permuted_values = graph.transpose(&new_values, &[1, 0, 2], None);

    let key_cache_slice_to_update = graph.slice(
        key_cache,
        1,
        prefix_length as i64,
        suffix_length as i64,
        None,
    );
    let value_cache_slice_to_update = graph.slice(
        &value_cache,
        1,
        prefix_length as i64,
        suffix_length as i64,
        None,
    );

    let assign_keys =
        graph.assign_variable(&key_cache_slice_to_update, &permuted_keys, None);
    let assign_values = graph.assign_variable(
        &value_cache_slice_to_update,
        &permuted_values,
        None,
    );

    let retained_graph_for_keys = graph.retain();
    let retained_key_cache = key_cache.retain();
    let keys = graph
        .control_dependency(
            &[&assign_keys],
            move || {
                vec![retained_graph_for_keys.slice(
                    &retained_key_cache,
                    1,
                    0,
                    (prefix_length + suffix_length) as i64,
                    None,
                )]
            },
            None,
        )
        .first()
        .unwrap()
        .clone();

    let retained_graph_for_values = graph.retain();
    let retained_value_cache = value_cache.retain();
    let values = graph
        .control_dependency(
            &[&assign_values],
            move || {
                vec![retained_graph_for_values.slice(
                    &retained_value_cache,
                    1,
                    0,
                    (prefix_length + suffix_length) as i64,
                    None,
                )]
            },
            None,
        )
        .first()
        .unwrap()
        .clone();

    let group_size = num_heads / num_groups;

    let expanded_keys = expand_keys_values(graph, &*keys, num_groups, head_dim);
    let expanded_values =
        expand_keys_values(graph, &*values, num_groups, head_dim);
    let expanded_queries = graph.expand_dims(&permuted_queries, 0, None);

    let scale = attention_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    let expanded_attention_output = graph
        .masked_scaled_dot_product_attention_with_scalar(
            &expanded_queries,
            &expanded_keys,
            &expanded_values,
            bias,
            scale,
            None,
        );

    let attention_output =
        graph.squeeze_axis(&expanded_attention_output, 0, None);
    let permuted_attention_output =
        graph.transpose(&attention_output, &[1, 0, 2], None);

    let output_shape =
        mps_shape(&[-1_i64, (num_groups * group_size * head_dim) as i64]);

    Ok(graph.reshape(&permuted_attention_output, &output_shape, None))
}
