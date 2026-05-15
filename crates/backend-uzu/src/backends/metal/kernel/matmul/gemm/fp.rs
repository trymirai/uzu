use super::{GemmDispatch, GemmKernel, GemmWeights};
use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::{
                GemmParams,
                gemm::{
                    GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTilingConfig,
                },
            },
            kernel::{
                TensorAddBiasKernel,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulError},
            },
        },
        metal::{Metal, context::MetalContext, kernel::TensorAddBiasMetalKernel, metal_extensions::DeviceExt},
    },
};

/// K-axis block size consumed per kernel iteration on the MXU path. The
/// `GemmTilingConfig::threadgroup_k` field on MXU tiles is the per-simdgroup
/// K-block, not this outer block size, so alignment must use this constant.
const MXU_BLOCK_K: u32 = 256;

pub(crate) fn encode(
    gemm: &mut GemmKernel,
    bias_add: &mut TensorAddBiasMetalKernel,
    data_type: DataType,
    context: &MetalContext,
    encoder: &mut Encoder<Metal>,
    arguments: MatmulArguments<Metal>,
    compute: GemmComputeKind,
) -> Result<(), MatmulError<Metal>> {
    if compute == GemmComputeKind::MxuMma && !context.device.supports_mxu() {
        return Err(MatmulError::UnsupportedDataType(data_type));
    }

    let tile = match compute {
        GemmComputeKind::SimdgroupMma => select_simdgroup_tile(data_type, &arguments),
        GemmComputeKind::MxuMma => select_mxu_tile(&arguments),
    };
    let k_divisor = match compute {
        GemmComputeKind::SimdgroupMma => tile.threadgroup_k,
        GemmComputeKind::MxuMma => MXU_BLOCK_K,
    };

    let threadgroups_per_row = arguments.output_dim.div_ceil(tile.threadgroup_n);
    let threadgroups_per_column = arguments.batch_dim.div_ceil(tile.threadgroup_m);

    let (use_morton, morton_total) = if compute == GemmComputeKind::MxuMma {
        let max_dim = threadgroups_per_row.max(threadgroups_per_column);
        let min_dim = threadgroups_per_row.min(threadgroups_per_column);
        let morton_dim = max_dim.next_power_of_two();
        let morton_total = morton_dim.saturating_mul(morton_dim);
        let actual_total = threadgroups_per_row.saturating_mul(threadgroups_per_column);
        let use_morton = min_dim > 1 && morton_total <= 4_u32.saturating_mul(actual_total);
        (use_morton, morton_total)
    } else {
        (false, 0)
    };
    let (group_count_x, group_count_y) = if use_morton {
        (morton_total, 1)
    } else {
        (threadgroups_per_row, threadgroups_per_column)
    };

    let alignment = GemmAlignment::from_axes(
        arguments.batch_dim % tile.threadgroup_m == 0,
        arguments.output_dim % tile.threadgroup_n == 0,
        arguments.input_dim % k_divisor == 0,
    );
    let output_transform = output_transform_from(&arguments);
    let mut params = build_params(&arguments, &tile);
    params.use_morton = use_morton;

    let dispatch = GemmDispatch {
        tiling_config: tile,
        input_prologue: GemmInputPrologueKind::FullPrecision,
        compute,
        output_transform,
        alignment,
        transpose_weights: arguments.b_transpose,
        weights: GemmWeights::FullPrecision {
            weights: arguments.b,
        },
        weights_offset: arguments.b_offset,
        activations: arguments.a,
        activations_offset: arguments.a_offset as usize,
        result: &mut *arguments.d,
        params,
        group_count_x,
        group_count_y,
    };

    gemm.encode(context, dispatch, encoder).map_err(MatmulError::BackendError)?;

    if let MatmulArgumentC::Bias(bias) = arguments.c {
        bias_add.encode(
            None::<&<Metal as crate::backends::common::Backend>::DenseBuffer>,
            bias,
            arguments.d,
            arguments.output_dim,
            arguments.batch_dim * arguments.output_dim,
            encoder,
        );
    }

    Ok(())
}

fn select_simdgroup_tile(
    data_type: DataType,
    arguments: &MatmulArguments<Metal>,
) -> GemmTilingConfig {
    let (threadgroup_m, threadgroup_n, threadgroup_k) = match data_type {
        DataType::F32 => (32u32, 64u32, 16u32),
        _ => {
            if 2 * arguments.batch_dim.max(arguments.output_dim) > arguments.input_dim {
                (64, 64, 16)
            } else {
                (64, 32, 32)
            }
        },
    };
    GemmTilingConfig {
        threadgroup_m,
        threadgroup_n,
        threadgroup_k,
        simdgroups_m: 2,
        simdgroups_n: 2,
    }
}

fn select_mxu_tile(arguments: &MatmulArguments<Metal>) -> GemmTilingConfig {
    let (threadgroup_m, threadgroup_n, simdgroups_m, simdgroups_n) =
        if arguments.batch_dim >= 256 && arguments.output_dim >= 128 {
            (128u32, 128u32, 4u32, 4u32)
        } else if arguments.output_dim < 64 {
            (64u32, 32u32, 4u32, 1u32)
        } else if arguments.batch_dim < 64 {
            (32u32, 64u32, 2u32, 2u32)
        } else {
            (64u32, 64u32, 2u32, 2u32)
        };
    GemmTilingConfig {
        threadgroup_m,
        threadgroup_n,
        threadgroup_k: 32,
        simdgroups_m,
        simdgroups_n,
    }
}

fn build_params(
    arguments: &MatmulArguments<Metal>,
    tile: &GemmTilingConfig,
) -> GemmParams {
    let default_ldb = if arguments.b_transpose {
        arguments.input_dim
    } else {
        arguments.output_dim
    };
    GemmParams {
        M: arguments.batch_dim,
        N: arguments.output_dim,
        K: arguments.input_dim,
        leading_dimension_activations: arguments.input_dim,
        leading_dimension_weights: arguments.b_leading_dimension.unwrap_or(default_ldb),
        leading_dimension_result: arguments.output_dim,
        threadgroups_per_row: arguments.output_dim.div_ceil(tile.threadgroup_n),
        threadgroups_per_column: arguments.batch_dim.div_ceil(tile.threadgroup_m),
        aligned_inner_iterations: arguments.input_dim / tile.threadgroup_k,
        ab_scale: arguments.ab_scale,
        ..Default::default()
    }
}

fn output_transform_from(arguments: &MatmulArguments<Metal>) -> GemmOutputTransformKind {
    let scale = arguments.ab_scale != 1.0;
    let accumulate = matches!(arguments.c, MatmulArgumentC::Accumulate);
    match (scale, accumulate) {
        (false, false) => GemmOutputTransformKind::Store,
        (true, false) => GemmOutputTransformKind::Scale,
        (false, true) => GemmOutputTransformKind::Accumulate,
        (true, true) => GemmOutputTransformKind::ScaleAccumulate,
    }
}
