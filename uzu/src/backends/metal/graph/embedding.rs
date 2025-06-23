#![allow(dead_code)]
use std::rc::Rc;

use mpsgraph::{Graph, GraphCallOps, GraphGatherOps, GraphMatrixOps, Tensor};
use objc2::rc::Retained;

use super::{
    super::MTLContext,
    GraphConstructionError,
    common::{load_constant, shaped_type},
};
use crate::{config::EmbeddingConfig, parameters::ParameterTree};

pub enum EmbeddingParams {
    Tied(Retained<Tensor>),
    Untied {
        input_weights: Retained<Tensor>,
        output_weights: Retained<Tensor>,
    },
}

pub fn embedding_params(
    graph: &Graph,
    config: &EmbeddingConfig,
    vocabulary_size: usize,
    model_dim: usize,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<EmbeddingParams, GraphConstructionError> {
    match config {
        EmbeddingConfig::Tied {
            common: _,
            precision,
        } => {
            let weights = load_constant(
                graph,
                &parameter_tree.subtree("embedding").unwrap(),
                "token_embeddings",
                &[vocabulary_size, model_dim],
                (*precision).into(),
            )?;
            Ok(EmbeddingParams::Tied(weights))
        },
        _ => {
            unimplemented!()
        },
    }
}

pub fn embed_callable_subgraph(
    graph: &Graph,
    config: &EmbeddingConfig,
    embeddings_callable_name: &str,
    embedding_dims: [isize; 2],
    input: &Tensor,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    match config {
        EmbeddingConfig::Tied {
            common: _,
            precision,
        } => {
            let embeddings_shaped_type =
                shaped_type(&embedding_dims, (*precision).into());
            if let Some(embeddings) = graph
                .call(
                    embeddings_callable_name,
                    &[],
                    &[&embeddings_shaped_type],
                    None,
                )
                .first()
            {
                let result = graph.gather(embeddings, input, 0, 0, None);
                Ok(result)
            } else {
                return Err(GraphConstructionError::CallableResultError {
                    name: embeddings_callable_name.to_string(),
                });
            }
        },
        _ => {
            unimplemented!()
        },
    }
}

pub fn embed_subgraph(
    graph: &Graph,
    config: &EmbeddingConfig,
    vocabulary_size: usize,
    model_dim: usize,
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    match config {
        EmbeddingConfig::Tied {
            common: _,
            precision,
        } => {
            let embeddings = load_constant(
                graph,
                &parameter_tree.subtree("embedding").unwrap(),
                "token_embeddings",
                &[vocabulary_size, model_dim],
                (*precision).into(),
            )
            .unwrap();
            let result = graph.gather(&embeddings, input, 0, 0, None);
            Ok(result)
        },
        _ => {
            unimplemented!()
        },
    }
}

pub fn embed_placeholder_weights_subgraph(
    graph: &Retained<Graph>,
    config: &EmbeddingConfig,
    input: &Retained<Tensor>,
    weights: &Retained<Tensor>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let result = graph.gather(weights, input, 0, 0, None);
    if let Some(scale) = config.common().input_scale {
        let scale_tensor =
            graph.constant_with_scalar(scale as f64, result.data_type());
        let scaled_result = graph.multiply(&result, &scale_tensor, None);
        Ok(scaled_result)
    } else {
        Ok(result)
    }
}

pub fn readout_callable_subgraph(
    graph: &Graph,
    config: &EmbeddingConfig,
    embeddings_callable_name: &str,
    embedding_dims: [isize; 2],
    input: &Tensor,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    match config {
        EmbeddingConfig::Tied {
            common: _,
            precision,
        } => {
            let embeddings_shaped_type =
                shaped_type(&embedding_dims, (*precision).into());
            if let Some(embeddings) = graph
                .call(
                    embeddings_callable_name,
                    &[],
                    &[&embeddings_shaped_type],
                    None,
                )
                .first()
            {
                let embeddings_transposed =
                    graph.transpose(embeddings, &[1, 0], None);
                let result = graph.matmul(
                    input,
                    &embeddings_transposed,
                    false,
                    false,
                    None,
                );
                Ok(result)
            } else {
                return Err(GraphConstructionError::CallableResultError {
                    name: embeddings_callable_name.to_string(),
                });
            }
        },
        _ => {
            unimplemented!()
        },
    }
}

pub fn readout_subgraph(
    graph: &Graph,
    config: &EmbeddingConfig,
    vocabulary_size: usize,
    model_dim: usize,
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    match config {
        EmbeddingConfig::Tied {
            common: _,
            precision,
        } => {
            let embeddings = load_constant(
                graph,
                &parameter_tree.subtree("embedding").unwrap(),
                "token_embeddings",
                &[vocabulary_size, model_dim],
                (*precision).into(),
            )
            .unwrap();
            let embeddings_transposed =
                graph.transpose(&embeddings, &[1, 0], None);
            let result =
                graph.matmul(input, &embeddings_transposed, false, false, None);
            Ok(result)
        },
        _ => {
            unimplemented!()
        },
    }
}

pub fn readout_placeholder_weights_subgraph(
    graph: &Retained<Graph>,
    input: &Retained<Tensor>,
    weights: &Retained<Tensor>,
    transpose_weights: bool,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    if transpose_weights {
        let embeddings_transposed = graph.transpose(weights, &[1, 0], None);
        let result =
            graph.matmul(input, &embeddings_transposed, false, false, None);
        Ok(result)
    } else {
        let result = graph.matmul(input, weights, false, false, None);
        Ok(result)
    }
}
