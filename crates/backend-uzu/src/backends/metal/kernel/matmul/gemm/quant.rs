use super::{GemmAlignmentAxes, GemmDispatch, GemmInputPrologueKind, GemmKernel, GemmWeights};
use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::{
                GemmParams, QuantizationMethod,
                gemm::{GemmAlignment, GemmOutputTransformKind, GemmTiling},
            },
            kernel::quant_matmul::{QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError},
        },
        metal::{Metal, context::MetalContext},
    },
};

pub(crate) fn encode(
    gemm: &mut GemmKernel,
    context: &MetalContext,
    configuration: &QuantizedMatmulConfiguration,
    arguments: QuantizedMatmulArguments<Metal>,
    encoder: &mut Encoder<Metal>,
) -> Result<(), QuantizedMatmulError<Metal>> {
    let tiling = select_tiling(configuration, arguments.batch_dim as u32);
    let group_size = configuration.group_size as u32;
    let batch_dim = arguments.batch_dim as u32;
    let input_dim = configuration.input_dim as u32;
    let output_dim = configuration.output_dim as u32;
    let threadgroups_per_row = output_dim.div_ceil(tiling.block_n());
    let threadgroups_per_column = batch_dim.div_ceil(tiling.block_m());
    let dispatch = GemmDispatch {
        tiling,
        input_prologue: GemmInputPrologueKind::FullPrecision,
        use_mxu: false,
        output_transform: GemmOutputTransformKind::Store,
        alignment: GemmAlignment::from_axes(GemmAlignmentAxes {
            m: batch_dim % tiling.block_m() == 0,
            n: output_dim % tiling.block_n() == 0,
            k: input_dim % tiling.block_k() == 0,
        }),
        transpose_b: true,
        a: arguments.a,
        a_offset: arguments.a_offset,
        b: match configuration.quantization_method {
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
        b_offset: 0,
        d: arguments.output,
        params: GemmParams {
            M: batch_dim,
            N: output_dim,
            K: input_dim,
            leading_dimension_a: input_dim,
            leading_dimension_b: input_dim,
            leading_dimension_d: output_dim,
            threadgroups_per_row,
            threadgroups_per_column,
            aligned_inner_iterations: input_dim / tiling.block_k(),
            use_morton: false,
            ab_scale: 1.0,
        },
        group_count_x: threadgroups_per_row,
        group_count_y: threadgroups_per_column,
    };
    gemm.encode(context, dispatch, encoder).map_err(QuantizedMatmulError::BackendError)
}

fn select_tiling(
    configuration: &QuantizedMatmulConfiguration,
    _batch_dim: u32,
) -> GemmTiling {
    let aligned_n_64 = configuration.output_dim % 64 == 0;
    let can_use_64_tiling = aligned_n_64 && configuration.data_type == DataType::BF16;

    if can_use_64_tiling {
        GemmTiling::T64x64x32_2x2
    } else {
        GemmTiling::T32x32x32_2x2
    }
}
