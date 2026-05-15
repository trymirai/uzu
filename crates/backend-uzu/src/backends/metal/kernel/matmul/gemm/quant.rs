use super::{GemmComputeKind, GemmDispatch, GemmInputPrologueKind, GemmKernel};
use crate::backends::{
    common::{
        Encoder,
        gpu_types::{
            GemmParams, QuantizationMethod,
            gemm::{GemmAlignment, GemmOutputTransformKind, GemmTilingConfig},
        },
        kernel::{
            gemm::GemmWeights,
            quant_matmul::{QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError},
        },
    },
    metal::{Metal, context::MetalContext},
};

pub(crate) fn encode(
    gemm: &mut GemmKernel,
    context: &MetalContext,
    configuration: &QuantizedMatmulConfiguration,
    arguments: QuantizedMatmulArguments<Metal>,
    encoder: &mut Encoder<Metal>,
) -> Result<(), QuantizedMatmulError<Metal>> {
    let tile = select_tile(configuration, arguments.batch_dim as u32);
    let group_size = configuration.group_size as u32;
    let batch_dim = arguments.batch_dim as u32;
    let input_dim = configuration.input_dim as u32;
    let output_dim = configuration.output_dim as u32;
    let group_count_x = output_dim.div_ceil(tile.threadgroup_n);
    let group_count_y = batch_dim.div_ceil(tile.threadgroup_m);
    let dispatch = GemmDispatch {
        tiling_config: tile,
        input_prologue: GemmInputPrologueKind::FullPrecision,
        compute: GemmComputeKind::SimdgroupMma,
        output_transform: GemmOutputTransformKind::Store,
        alignment: GemmAlignment::from_axes(
            batch_dim % tile.threadgroup_m == 0,
            output_dim % tile.threadgroup_n == 0,
            input_dim % tile.threadgroup_k == 0,
        ),
        weights: match configuration.quantization_method {
            QuantizationMethod::ScaleBias => GemmWeights::ScaleBias {
                weights: arguments.b,
                scales: arguments.scales,
                biases: arguments.zero_points_or_biases,
                mode: configuration.mode,
                group_size,
            },
            QuantizationMethod::ScaleZeroPoint => GemmWeights::ScaleZeroPoint {
                weights: arguments.b,
                scales: arguments.scales,
                zero_points: arguments.zero_points_or_biases,
                mode: configuration.mode,
                group_size,
            },
        },
        activations: arguments.a,
        activations_offset: arguments.a_offset,
        result: arguments.output,
        params: GemmParams {
            M: batch_dim,
            N: output_dim,
            K: input_dim,
            leading_dimension_a: input_dim,
            leading_dimension_b: input_dim,
            leading_dimension_d: output_dim,
            threadgroups_per_row: group_count_x,
            threadgroups_per_column: group_count_y,
            aligned_inner_iterations: input_dim / tile.threadgroup_k,
            ..Default::default()
        },
        group_count_x,
        group_count_y,
    };
    gemm.encode(context, dispatch, encoder).map_err(QuantizedMatmulError::BackendError)
}

fn select_tile(
    configuration: &QuantizedMatmulConfiguration,
    batch_dim: u32,
) -> GemmTilingConfig {
    let group_size = configuration.group_size as u32;
    let threadgroup_k = group_size.min(32);
    let (threadgroup_m, threadgroup_n, simdgroups_m, simdgroups_n) = if batch_dim < 32 {
        (32u32, 32u32, 1u32, 1u32)
    } else {
        (64u32, 64u32, 2u32, 2u32)
    };
    GemmTilingConfig {
        threadgroup_m,
        threadgroup_n,
        threadgroup_k,
        simdgroups_m,
        simdgroups_n,
    }
}
