use std::rc::Rc;

use mpsgraph::{Graph, GraphMatrixOps, Tensor};
use objc2::rc::Retained;

use super::{super::MTLContext, GraphConstructionError, load_constant};
use crate::{config::LinearConfig, parameters::ParameterTree};

pub fn linear_subgraph<const N: usize>(
    graph: &Graph,
    config: &LinearConfig,
    input_dim: usize,
    output_dims: [usize; N],
    has_biases: bool,
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    match config {
        LinearConfig::FullPrecision {
            precision,
        } => {
            let output_dim_sum: usize = output_dims.iter().sum();
            let weights = load_constant(
                graph,
                parameter_tree,
                "weights",
                &[input_dim, output_dim_sum],
                (*precision).into(),
            )?;

            let matmul = graph.transpose(
                &graph.matmul(
                    &graph.transpose(&weights, &[1, 0], None),
                    &graph.transpose(input, &[1, 0], None),
                    false,
                    false,
                    None,
                ),
                &[1, 0],
                None,
            );

            if has_biases {
                let biases = load_constant(
                    graph,
                    parameter_tree,
                    "biases",
                    &[output_dim_sum],
                    (*precision).into(),
                )?;

                let result = graph.add(&matmul, &biases, None);

                Ok(result)
            } else {
                Ok(matmul)
            }
        },
        LinearConfig::Quantized {
            ..
        } => {
            // Quantized linear layer implementation
            unimplemented!(
                "Quantized linear layer implementation not yet available"
            )
        },
        LinearConfig::QLoRA {
            ..
        } => {
            // QLoRA linear layer implementation
            unimplemented!(
                "QLoRA linear layer implementation not yet available"
            )
        },
    }
}
