use std::rc::Rc;

use mpsgraph::{Graph, Tensor};
use objc2::rc::Retained;

use super::{
    super::MTLContext, GraphConstructionError, common::activation,
    linear::linear_subgraph,
};
use crate::{config::MLPConfig, parameters::ParameterTree};

pub fn mlp_subgraph(
    graph: &Graph,
    config: &MLPConfig,
    model_dim: usize,
    hidden_dim: usize,
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let up_tree = parameter_tree.subtree("up_projection")?;
    let down_tree = parameter_tree.subtree("down_projection")?;

    let fused_hidden_gate = linear_subgraph(
        graph,
        &config.linear_config,
        model_dim,
        [hidden_dim, hidden_dim],
        false,
        input,
        &up_tree,
    )?;

    let split_results =
        graph.split_tensor_num_splits(&fused_hidden_gate, 2, 1, None);
    let up_proj = &split_results[0];
    let gate = &split_results[1];

    let activated_gate = activation(
        graph,
        &config.activation,
        &*gate,
        config.linear_config.activation_precision().into(),
    );
    let hidden = graph.multiplication(&activated_gate, up_proj, None);

    let result = linear_subgraph(
        graph,
        &config.linear_config,
        hidden_dim,
        [model_dim],
        false,
        &*hidden,
        &down_tree,
    )?;

    Ok(result)
}
