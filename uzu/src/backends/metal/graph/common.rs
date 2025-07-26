use std::rc::Rc;

use mpsgraph::{Graph, Shape, ShapedType, Tensor};
use objc2::rc::Retained;
use thiserror::Error;

use super::super::utils::mps_shape;
use crate::{
    Array, DataType,
    backends::metal::{MTLContext, MetalArray},
    config::Activation,
    parameters::{ParameterLoaderError, ParameterTree},
};

pub fn mps_shape_to_isize(shape: &Shape) -> Box<[isize]> {
    let shape_i64: Box<[i64]> = *shape.into();
    shape_i64.iter().map(|&d| d as isize).collect()
}

pub fn shape_of(tensor: &Tensor) -> Box<[isize]> {
    mps_shape_to_isize(&tensor.shape())
}

pub fn data_type_of(tensor: &Tensor) -> DataType {
    DataType::from(tensor.data_type())
}

#[derive(Debug, Error)]
pub enum GraphConstructionError {
    #[error("Failed to load weights: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError),
    #[error(
        "Incompatible shapes. Node {node_name} at subtree {node_path:?} expected to have shape {expected:?} but got {actual:?}"
    )]
    IncompatibleShapes {
        node_path: Option<String>,
        node_name: String,
        expected: Box<[usize]>,
        actual: Box<[usize]>,
    },
    #[error(
        "Incompatible data types. Node {node_name} at subtree {node_path:?} expected to have data type {expected:?} but got {actual:?}"
    )]
    IncompatibleDataTypes {
        node_path: Option<String>,
        node_name: String,
        expected: DataType,
        actual: DataType,
    },
    #[error("Failed to run callable")]
    CallableResultError {
        name: String,
    },
}

pub fn activation(
    graph: &Graph,
    config: &Activation,
    input: &Tensor,
    data_type: DataType,
) -> Retained<Tensor> {
    match config {
        Activation::SILU => {
            let sigmoid_result = graph.sigmoid(input, Some("sigmoid"));
            graph.multiplication(input, &sigmoid_result, Some("silu_output"))
        },
        Activation::GELU => gelu(graph, input, data_type, true),
    }
}

pub fn gelu(
    graph: &Graph,
    input: &Tensor,
    data_type: DataType,
    approximate: bool,
) -> Retained<Tensor> {
    if approximate {
        let input_power_2 = graph.multiplication(input, input, None);
        let input_power_3 = graph.multiplication(&input_power_2, input, None);
        let input_power_3_multiply_const = graph.multiplication(
            &graph.constant_with_scalar(
                0.044715 as f64,
                None,
                Some(data_type.into()),
            ),
            &input_power_3,
            None,
        );
        let input_add_input_power_3_multiply_const =
            graph.addition(input, &input_power_3_multiply_const, None);
        let tanh = GraphActivationOps::tanh(
            graph,
            &graph.multiplication(
                &graph.constant_with_scalar(
                    (2.0 / std::f64::consts::PI).sqrt(),
                    Some(data_type.into()),
                ),
                &input_add_input_power_3_multiply_const,
                None,
            ),
            None,
        );
        let tanh_add_one = graph.addition(
            &graph.constant_with_scalar(1.0 as f64, Some(data_type.into())),
            &tanh,
            None,
        );
        let input_multiply_tanh_add_one =
            graph.multiplication(input, &tanh_add_one, None);
        let result = graph.multiplication(
            &input_multiply_tanh_add_one,
            &graph.constant_with_scalar(0.5 as f64, Some(data_type.into())),
            None,
        );
        result
    } else {
        let input_negative = graph.multiplication(
            &graph.constant_with_scalar(-1.0 as f64, Some(data_type.into())),
            input,
            None,
        );
        let input_negative_divide_sqrt_2 = graph.division(
            &input_negative,
            &graph.constant_with_scalar(
                (2.0 as f64).sqrt(),
                Some(data_type.into()),
            ),
            None,
        );
        let erf = graph.erf(&input_negative_divide_sqrt_2, None);
        let erf_negative = graph.multiplication(
            &graph.constant_with_scalar(-1.0 as f64, data_type.into()),
            &erf,
            None,
        );
        let erfc = graph.addition(
            &graph.constant_with_scalar(1.0 as f64, Some(data_type.into())),
            &erf_negative,
            None,
        );
        let input_divide_2 = graph.division(
            &input,
            &graph.constant_with_scalar(
                2.0 as f64,
                Some(data_type.into()),
                None,
            ),
            None,
        );
        let result = graph.multiplication(&input_divide_2, &erfc, None);
        result
    }
}

pub fn placeholder(
    graph: &Graph,
    shape: &[isize],
    data_type: DataType,
) -> Retained<Tensor> {
    let shape_i64: Box<[i64]> = shape.iter().map(|dim| *dim as i64).collect();
    graph.placeholder(
        Some(&Shape::from_dimensions(&shape_i64)),
        Some(data_type.into()),
        Some("input"),
    )
}

pub fn shaped_type(
    shape: &[isize],
    data_type: DataType,
) -> Retained<ShapedType> {
    let shape_i64: Box<[i64]> = shape.iter().map(|dim| *dim as i64).collect();
    ShapedType::new(&Shape::from_dimensions(&shape_i64), data_type.into())
}

fn map_last_dimension<F: FnMut(usize) -> usize>(
    mut f: F,
    shape: &[usize],
) -> Box<[usize]> {
    shape
        .iter()
        .enumerate()
        .map(|(i, dim)| {
            if i == shape.len() - 1 {
                f(*dim)
            } else {
                *dim
            }
        })
        .collect()
}

fn unpacked_shape_from_packed_shape(packed_shape: &[usize]) -> Box<[usize]> {
    map_last_dimension(|dim| dim * 2, packed_shape)
}

fn packed_shape_from_unpacked_shape(packed_shape: &[usize]) -> Box<[usize]> {
    map_last_dimension(|dim| dim / 2, packed_shape)
}

pub fn i4_constant_from_packed_u8_array(
    graph: &Graph,
    data: &MetalArray,
) -> Retained<Tensor> {
    assert!(data.data_type() == DataType::U8);
    graph.constant_with_data(
        data.buffer(),
        &mps_shape(&unpacked_shape_from_packed_shape(data.shape())),
        DataType::I4.into(),
    )
}

pub fn load_constant(
    graph: &Graph,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
    name: &str,
    expected_shape: &[usize],
    expected_data_type: DataType,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let parameter = parameter_tree.leaf(name)?;
    let result = if expected_data_type == DataType::U4 {
        if parameter.data_type() != DataType::U8 {
            return Err(GraphConstructionError::IncompatibleDataTypes {
                node_path: parameter_tree.path_prefix().map(str::to_string),
                node_name: name.to_string(),
                expected: DataType::U8,
                actual: parameter.data_type(),
            });
        };
        let packed_expected_shape =
            packed_shape_from_unpacked_shape(expected_shape);
        if parameter.shape() != &*packed_expected_shape {
            return Err(GraphConstructionError::IncompatibleShapes {
                node_path: parameter_tree.path_prefix().map(str::to_string),
                node_name: name.to_string(),
                expected: packed_expected_shape.into(),
                actual: parameter.shape().into(),
            });
        };
        graph.constant_with_data(
            parameter.buffer(),
            &mps_shape(expected_shape),
            DataType::U4.into(),
        )
    } else {
        if parameter.shape() != expected_shape {
            return Err(GraphConstructionError::IncompatibleShapes {
                node_path: parameter_tree.path_prefix().map(str::to_string),
                node_name: name.to_string(),
                expected: expected_shape.into(),
                actual: parameter.shape().into(),
            });
        }
        if parameter.data_type() != expected_data_type {
            return Err(GraphConstructionError::IncompatibleDataTypes {
                node_path: parameter_tree.path_prefix().map(str::to_string),
                node_name: name.to_string(),
                expected: expected_data_type,
                actual: parameter.data_type(),
            });
        };
        graph.constant_with_data(
            parameter.buffer(),
            &mps_shape(parameter.shape()),
            parameter.data_type().into(),
        )
    };
    Ok(result)
}
