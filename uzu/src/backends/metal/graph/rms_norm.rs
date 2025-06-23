use std::rc::Rc;

use mpsgraph::{
    Graph, GraphArithmeticOps, GraphNormalizationOps, Tensor,
    tensor_shape_ops::*,
};
use objc2::rc::Retained;

use super::{
    super::MTLContext, GraphConstructionError, data_type_of, load_constant,
    shape_of,
};
use crate::{
    DataType,
    config::{RMSNormConfig, UpcastMode},
    parameters::ParameterTree,
};

fn cast_if_needed(
    graph: &Graph,
    input: &Retained<Tensor>,
    data_type: DataType,
) -> Retained<Tensor> {
    if data_type_of(input) != data_type {
        graph.cast(input, data_type.into(), None)
    } else {
        input.clone()
    }
}

fn adjusted_variance(
    graph: &Graph,
    input: &Retained<Tensor>,
    epsilon: f32,
) -> Retained<Tensor> {
    let epsilon_constant =
        graph.constant_with_scalar(epsilon as f64, input.data_type());

    let input_squared = graph.square(&input, None);
    let input_squared_mean = graph.mean(&input_squared, &[1], None);
    let input_squared_mean_adjusted =
        graph.add(&input_squared_mean, &epsilon_constant, None);

    input_squared_mean_adjusted
}

fn normalization(
    graph: &Graph,
    input: &Retained<Tensor>,
    adjusted_variance: &Retained<Tensor>,
) -> Retained<Tensor> {
    let inverse_rms_norm = graph.rsqrt(&adjusted_variance, None);
    let normalized_input = graph.multiply(input, &inverse_rms_norm, None);
    normalized_input
}

pub fn rms_norm_subgraph(
    graph: &Graph,
    config: &RMSNormConfig,
    input: &Retained<Tensor>,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let [_, model_dim] = *shape_of(input) else {
        panic!(
            "Expected two-dimensional input, received {0:?}",
            shape_of(input)
        );
    };
    assert_ne!(model_dim, -1);

    let accumulation_data_type: DataType = config.accumulation_precision.into();
    let scale_data_type: DataType = config.scale_precision.into();

    let x = cast_if_needed(graph, input, accumulation_data_type);
    let adjusted_variance = adjusted_variance(graph, &x, config.epsilon);
    let mut normalized_x = normalization(graph, &x, &adjusted_variance);
    match config.upcast_mode {
        UpcastMode::OnlyNormalization => {
            normalized_x =
                cast_if_needed(graph, &normalized_x, scale_data_type);
        },
        UpcastMode::FullLayer => {},
    }

    let mut scales = load_constant(
        graph,
        parameter_tree,
        "scales",
        &[model_dim as usize],
        scale_data_type,
    )?;
    match config.upcast_mode {
        UpcastMode::OnlyNormalization => {},
        UpcastMode::FullLayer => {
            scales = cast_if_needed(graph, &scales, accumulation_data_type);
        },
    }
    if let Some(scale_offset) = config.scale_offset {
        let scale_offset_constant =
            graph.constant_with_scalar(scale_offset as f64, scales.data_type());
        scales = graph.add(&scales, &scale_offset_constant, None);
    }

    let mut result = graph.multiply(&normalized_x, &scales, None);
    result = cast_if_needed(graph, &result, scale_data_type);
    assert_eq!(shape_of(&*result), shape_of(input));
    Ok(result)
}
