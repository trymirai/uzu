use mpsgraph::{DequantizationArguments, Graph, Tensor};
use objc2::rc::Retained;

use super::GraphConstructionError;
use crate::config::EmbeddingConfig;

pub fn embeddings_dequantize_weights_subgraph(
    graph: &Retained<Graph>,
    weights: &Retained<Tensor>,
    scales: &Retained<Tensor>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let result = graph.dequantize(
        &weights,
        DequantizationArguments::ScaleTensorZeroPointDataTypeAxis {
            scale_tensor: &scales,
            zero_point: 0.0,
            data_type: scales.data_type(),
            axis: 0,
        },
        None,
    );
    Ok(result)
}

pub fn embeddings_embed_subgraph(
    graph: &Retained<Graph>,
    config: &EmbeddingConfig,
    input: &Retained<Tensor>,
    weights: &Retained<Tensor>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let result = graph.gather_with_updates(&weights, input, 0, 0, None);
    if let Some(scale) = config.common().input_scale {
        let scale_tensor =
            graph.constant_with_scalar(scale as f64, None, result.data_type());
        let scaled_result = graph.multiplication(&result, &scale_tensor, None);
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
    let result = graph.matrix_multiplication(input, weights, None);
    Ok(result)
}
