use crate::{
    backends::common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE,
    config::weight_matrix::{AnyWeightMatrixSpec, Layout, int_spec::IntSpec, mlx_spec::MLXSpec},
    data_type::DataType,
};

pub const ACTIVATION_QUANTIZATION_GROUP_SIZE: u32 = 32;

fn supported_weight_group_size(group_size: u32) -> bool {
    matches!(group_size, 32 | 64 | 128)
}

fn weight_bits_and_group_size(spec: &AnyWeightMatrixSpec) -> Option<(u32, u32)> {
    match spec {
        AnyWeightMatrixSpec::IntSpec(IntSpec {
            bits,
            group_size,
            layout: Layout::OutputInput,
            ..
        })
        | AnyWeightMatrixSpec::MLXSpec(MLXSpec {
            bits,
            group_size,
            layout: Layout::OutputInput,
            ..
        }) => Some((*bits, u32::try_from(*group_size).ok()?)),
        _ => None,
    }
}

pub fn activation_quantization_group_size_for_rht_linear(
    supports_symmetric_int8_activations: bool,
    input_dimension: usize,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    quantization_spec: &AnyWeightMatrixSpec,
) -> Option<u32> {
    if !supports_symmetric_int8_activations
        || [weights_data_type, input_data_type, output_data_type].into_iter().any(|dt| dt != DataType::BF16)
    {
        return None;
    }
    let (bits, group_size) = weight_bits_and_group_size(quantization_spec)?;
    (matches!(bits, 4 | 8)
        && supported_weight_group_size(group_size)
        && input_dimension.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)
        && input_dimension.is_multiple_of(group_size as usize))
    .then_some(ACTIVATION_QUANTIZATION_GROUP_SIZE)
}
