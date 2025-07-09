use std::rc::Rc;

use mpsgraph::{
    Graph, GraphMatrixOps, GraphQuantizationOps, GraphTensorShapeOps, Tensor,
};
use objc2::rc::Retained;

use super::{super::MTLContext, GraphConstructionError, load_constant};
use crate::{
    DataType,
    config::{
        ConfigDataType, LinearConfig, QuantizationConfig, QuantizationMode,
    },
    parameters::ParameterTree,
};

fn full_precision_weights_subgraph<const N: usize>(
    graph: &Graph,
    precision: DataType,
    input_dim: usize,
    output_dims: [usize; N],
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let output_dim_sum: usize = output_dims.iter().sum();
    load_constant(
        graph,
        parameter_tree,
        "weights",
        &[input_dim, output_dim_sum],
        precision,
    )
}

fn quantized_weights_subgraph<const N: usize>(
    graph: &Graph,
    config: &QuantizationConfig,
    input_dim: usize,
    output_dims: [usize; N],
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    if config.weight_quantization_mode != QuantizationMode::UInt4 {
        unimplemented!("8-bit quantization is not yet implemented.");
    }
    let output_dim_sum: usize = output_dims.iter().sum();

    let weights = load_constant(
        graph,
        parameter_tree,
        "weights",
        &[input_dim, output_dim_sum],
        DataType::U4,
    )?;

    let scales = load_constant(
        graph,
        parameter_tree,
        "scales",
        &[input_dim, output_dim_sum / config.group_size],
        config.activation_precision.into(),
    )?;

    let zero_points = load_constant(
        graph,
        parameter_tree,
        "zero_points",
        &[input_dim, output_dim_sum / config.group_size],
        DataType::U4,
    )?;

    Ok(graph
        .dequantize_with_scale_tensor_and_zero_point_tensor(
            &weights,
            &scales,
            &zero_points,
            <ConfigDataType as Into<DataType>>::into(
                config.activation_precision,
            )
            .into(),
            None,
        )
        .unwrap())
}

fn biases_subgraph<const N: usize>(
    graph: &Graph,
    precision: DataType,
    output_dims: [usize; N],
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let output_dim_sum: usize = output_dims.iter().sum();
    load_constant(graph, parameter_tree, "biases", &[output_dim_sum], precision)
}

struct LoraWeights {
    down_weights: Retained<Tensor>,
    up_weights: Box<[Retained<Tensor>]>,
}

fn lora_weights_subgraph<const N: usize>(
    graph: &Graph,
    precision: DataType,
    input_dim: usize,
    output_dims: [usize; N],
    lora_rank: usize,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<LoraWeights, GraphConstructionError> {
    let down_weights = load_constant(
        graph,
        parameter_tree,
        "down_weights",
        &[input_dim, lora_rank * N],
        precision,
    )?;
    let up_weights = output_dims
        .iter()
        .enumerate()
        .map(|(i, output_dim)| {
            load_constant(
                graph,
                parameter_tree,
                &format!("up_weights.{}", i),
                &[lora_rank, *output_dim],
                precision,
            )
        })
        .collect::<Result<Box<[_]>, _>>()?;
    Ok(LoraWeights {
        down_weights,
        up_weights: up_weights,
    })
}

fn full_precision_matmul_subgraph<const N: usize>(
    graph: &Graph,
    precision: DataType,
    input_dim: usize,
    output_dims: [usize; N],
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let weights = full_precision_weights_subgraph::<N>(
        graph,
        precision,
        input_dim,
        output_dims,
        parameter_tree,
    )?;
    Ok(graph.transpose(
        &graph.matmul(
            &graph.transpose(&weights, &[1, 0], None),
            &graph.transpose(input, &[1, 0], None),
            None,
        ),
        &[1, 0],
        None,
    ))
}

fn quantized_matmul_subgraph<const N: usize>(
    graph: &Graph,
    config: &QuantizationConfig,
    input_dim: usize,
    output_dims: [usize; N],
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let weights = quantized_weights_subgraph::<N>(
        graph,
        config,
        input_dim,
        output_dims,
        parameter_tree,
    )?;
    Ok(graph.matmul(input, &weights, None))
}

fn lora_subgraph<const N: usize>(
    graph: &Graph,
    precision: DataType,
    lora_rank: usize,
    input_dim: usize,
    output_dims: [usize; N],
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let lora_weights = lora_weights_subgraph(
        graph,
        precision,
        input_dim,
        output_dims,
        lora_rank,
        parameter_tree,
    )?;

    let lora_inner = graph.transpose(
        &graph.matmul(
            &graph.transpose(&lora_weights.down_weights, &[1, 0], None),
            &graph.transpose(input, &[1, 0], None),
            None,
        ),
        &[1, 0],
        None,
    );

    let lora_inner_splits =
        graph.split(&lora_inner, output_dims.len(), 1, None);
    assert_eq!(lora_inner_splits.len(), lora_weights.up_weights.len());

    let lora_outputs = lora_inner_splits
        .iter()
        .zip(lora_weights.up_weights.iter())
        .map(|(lora_inner_split, up_weights)| {
            graph.transpose(
                &graph.matmul(
                    &graph.transpose(&up_weights, &[1, 0], None),
                    &graph.transpose(&lora_inner_split, &[1, 0], None),
                    None,
                ),
                &[1, 0],
                None,
            )
        })
        .collect::<Box<[Retained<Tensor>]>>();

    Ok(graph.concat_tensors(
        &lora_outputs
            .iter()
            .map(|tensor| &**tensor)
            .collect::<Box<[&Tensor]>>(),
        1,
        None,
    ))
}

fn qlora_matmul_subgraph<const N: usize>(
    graph: &Graph,
    lora_rank: usize,
    lora_scale: f32,
    config: &QuantizationConfig,
    input_dim: usize,
    output_dims: [usize; N],
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let quantized_matmul_result = quantized_matmul_subgraph::<N>(
        graph,
        config,
        input_dim,
        output_dims,
        input,
        parameter_tree,
    )?;
    let lora_result = lora_subgraph::<N>(
        graph,
        config.activation_precision.into(),
        lora_rank,
        input_dim,
        output_dims,
        input,
        parameter_tree,
    )?;
    let scaled_lora_result = graph.multiply(
        &lora_result,
        &graph.constant_with_scalar(
            lora_scale as f64,
            <ConfigDataType as Into<DataType>>::into(
                config.activation_precision,
            )
            .into(),
        ),
        None,
    );
    Ok(graph.add(&quantized_matmul_result, &scaled_lora_result, None))
}

pub fn linear_subgraph<const N: usize>(
    graph: &Graph,
    config: &LinearConfig,
    input_dim: usize,
    output_dims: [usize; N],
    has_biases: bool,
    input: &Tensor,
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<Retained<Tensor>, GraphConstructionError> {
    let matmul_result = match config {
        LinearConfig::FullPrecision {
            precision,
        } => full_precision_matmul_subgraph(
            graph,
            (*precision).into(),
            input_dim,
            output_dims,
            input,
            parameter_tree,
        )?,
        LinearConfig::Quantized(quantization_config) => {
            quantized_matmul_subgraph(
                graph,
                quantization_config,
                input_dim,
                output_dims,
                input,
                parameter_tree,
            )?
        },
        LinearConfig::QLoRA {
            quantization: quantization_config,
            lora_rank,
            lora_scale,
        } => qlora_matmul_subgraph(
            graph,
            *lora_rank,
            *lora_scale,
            quantization_config,
            input_dim,
            output_dims,
            input,
            parameter_tree,
        )?,
    };
    if has_biases {
        let biases = biases_subgraph(
            graph,
            config.activation_precision().into(),
            output_dims,
            parameter_tree,
        )?;
        Ok(graph.add(&matmul_result, &biases, None))
    } else {
        Ok(matmul_result)
    }
}
