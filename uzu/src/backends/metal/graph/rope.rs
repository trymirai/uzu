use std::rc::Rc;

use mpsgraph::{Graph, GraphGatherOps, Tensor, tensor_shape_ops::*};
use objc2::rc::Retained;

use super::{
    super::MTLContext, GraphConstructionError, common::data_type_of,
    load_constant, shape_of,
};
use crate::{DataType, config::RoPEConfig, parameters::ParameterTree};

pub struct PositionalEmbeddings {
    sines: Retained<Tensor>,
    cosines: Retained<Tensor>,
}

impl PositionalEmbeddings {
    pub fn new(
        sines: Retained<Tensor>,
        cosines: Retained<Tensor>,
    ) -> Self {
        let cosines_shape = shape_of(&*cosines);
        let sines_shape = shape_of(&*sines);
        assert_eq!(cosines_shape, sines_shape);
        Self {
            sines,
            cosines,
        }
    }
}

pub fn positional_embeddings_subgraph(
    graph: &Graph,
    rope_config: &RoPEConfig,
    head_dim: usize,
    context_length: usize,
    indices: &Tensor, // u64: [max_sequence_length]
    parameter_tree: &ParameterTree<Rc<MTLContext>>,
) -> Result<PositionalEmbeddings, GraphConstructionError> {
    assert_eq!(data_type_of(indices), DataType::I32);
    let expected_shape = [context_length, head_dim];
    let expected_data_type: DataType = rope_config.common().precision.into();
    let all_cosines = load_constant(
        graph,
        parameter_tree,
        "cosines",
        &expected_shape,
        expected_data_type,
    )?;
    let all_sines = load_constant(
        graph,
        parameter_tree,
        "sines",
        &expected_shape,
        expected_data_type,
    )?;

    let cosine_slice = graph.gather(&all_cosines, indices, 0, 0, None);
    let sine_slice = graph.gather(&all_sines, indices, 0, 0, None);
    Ok(PositionalEmbeddings::new(sine_slice, cosine_slice))
}

pub fn apply_rope_subgraph(
    graph: &Graph,
    positional_embeddings: &PositionalEmbeddings,
    input: &Tensor,
) -> Retained<Tensor> {
    let [suffix_length, _, head_dim] = *shape_of(input) else {
        panic!(
            "Expected three-dimensional input tensor, got {0:?}",
            shape_of(input)
        )
    };
    let pe_shape = shape_of(&*positional_embeddings.cosines);
    let [pe_suffix_length, pe_head_dim] = *pe_shape else {
        panic!(
            "Expected two-dimensional positional embeddings, got {0:?}",
            pe_shape
        )
    };
    assert_eq!(head_dim, pe_head_dim);
    assert_eq!(suffix_length, pe_suffix_length);

    let x = graph.slice(input, 2, 0, (head_dim / 2) as i64, None);
    let y = graph.slice(
        input,
        2,
        (head_dim / 2) as i64,
        (head_dim / 2) as i64,
        None,
    );
    // [suffix_length, num_heads, head_dim / 2]
    let neg_y = graph.multiply(
        &y,
        &graph.constant_with_scalar(-1.0_f64, y.data_type()),
        None,
    );
    // [suffix_length, num_heads, head_dim]
    let half_rotated_heads = graph.concat(&neg_y, &x, 2, None);
    // [suffix_length, 1, head_dim]
    let unsqueezed_cosines = graph.expand_dims(
        &positional_embeddings.cosines,
        1,
        Some("unsqueezed_cosines_expand_dims"),
    );
    // [suffix_length, 1, head_dim]
    let unsqueezed_sines = graph.expand_dims(
        &positional_embeddings.sines,
        1,
        Some("unsqueezed_sines_expand_dims"),
    );
    // [suffix_length, num_heads, head_dim]
    let mul_cos = graph.multiply(input, &unsqueezed_cosines, None);
    let mul_sin = graph.multiply(&half_rotated_heads, &unsqueezed_sines, None);
    // [suffix_length, num_heads, head_dim]
    let result = graph.add(&mul_cos, &mul_sin, None);

    assert_eq!(shape_of(input), shape_of(&*result));

    result
}
