use std::rc::Rc;

use mpsgraph::{Graph, GraphQuantizationOps, GraphTensorShapeOps, Tensor};
use objc2::rc::Retained;

use super::{super::MTLContext, GraphConstructionError, load_constant};
use crate::DataType;
use crate::config::ConfigDataType;
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
        LinearConfig::Quantized(quantization_config) => {
            let output_dim_sum: usize = output_dims.iter().sum();
            let group_size = quantization_config.group_size;
            let activation_precision = quantization_config.activation_precision;

            let weights = load_constant(
                graph,
                parameter_tree,
                "weights",
                &[output_dim_sum, input_dim],
                DataType::U4,
            )?;

            let scales = load_constant(
                graph,
                parameter_tree,
                "scales",
                &[output_dim_sum, input_dim / group_size],
                activation_precision.into(),
            )?;

            let zero_points = load_constant(
                graph,
                parameter_tree,
                "zero_points",
                &[output_dim_sum, input_dim / group_size],
                DataType::U4,
            )?;

            let dequantized_weights = graph
                .dequantize_with_scale_tensor_and_zero_point_tensor(
                    &weights,
                    &scales,
                    &zero_points,
                    <ConfigDataType as Into<DataType>>::into(
                        activation_precision,
                    )
                    .into(),
                    None,
                )
                .unwrap();

            let matmul = graph.matmul(
                &input,
                &graph.transpose(&dequantized_weights, &[1, 0], None),
                false,
                false,
                None,
            );

            if has_biases {
                let biases = load_constant(
                    graph,
                    parameter_tree,
                    "biases",
                    &[output_dim_sum],
                    activation_precision.into(),
                )?;

                let result = graph.add(&matmul, &biases, None);

                Ok(result)
            } else {
                Ok(matmul)
            }
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
