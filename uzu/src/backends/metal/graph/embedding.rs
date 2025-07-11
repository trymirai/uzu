use mpsgraph::{
    Graph, GraphGatherOps, GraphMatrixOps, GraphQuantizationOps, Tensor,
};
use objc2::rc::Retained;

use super::GraphConstructionError;
use crate::config::EmbeddingConfig;

pub fn embeddings_dequantize_weights_subgraph(
    graph: &Retained<Graph>,
    weights: &Retained<Tensor>,
    scales: &Retained<Tensor>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let result = graph
        .dequantize_with_scale_tensor_zero_point_and_axis(
            &weights,
            &scales,
            0.0,
            scales.data_type(),
            0,
            None,
        )
        .unwrap();
    Ok(result)
}

pub fn embeddings_embed_subgraph(
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

pub fn embeddings_readout_subgraph(
    graph: &Retained<Graph>,
    input: &Retained<Tensor>,
    weights: &Retained<Tensor>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let result = graph.matmul(input, weights, None);
    Ok(result)
}
