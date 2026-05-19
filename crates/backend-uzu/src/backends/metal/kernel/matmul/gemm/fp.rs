use super::{GemmAlignmentAxes, GemmDispatch, GemmKernel, GemmWeights, MXU_THREADGROUP_BLOCK_K};
use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::{
                GemmParams,
                gemm::{GemmAlignment, GemmInputPrologueKind, GemmOutputTransformKind, GemmTiling},
            },
            kernel::{
                TensorAddBiasKernel,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulError},
            },
        },
        metal::{Metal, context::MetalContext, kernel::TensorAddBiasMetalKernel, metal_extensions::DeviceExt},
    },
};

pub(crate) fn encode(
    gemm: &mut GemmKernel,
    bias_add: &mut TensorAddBiasMetalKernel,
    data_type: DataType,
    context: &MetalContext,
    encoder: &mut Encoder<Metal>,
    arguments: MatmulArguments<Metal>,
    use_mxu: bool,
) -> Result<(), MatmulError<Metal>> {
    if use_mxu && !context.device.supports_mxu() {
        return Err(MatmulError::UnsupportedDataType(data_type));
    }

    let tiling = if use_mxu {
        select_mxu_tiling(&arguments)
    } else {
        select_simdgroup_tiling(&arguments)
    };
    let k_block = if use_mxu {
        MXU_THREADGROUP_BLOCK_K
    } else {
        tiling.block_k()
    };

    let threadgroups_per_row = arguments.output_dim.div_ceil(tiling.block_n());
    let threadgroups_per_column = arguments.batch_dim.div_ceil(tiling.block_m());

    let (use_morton, group_count_x, group_count_y) = if use_mxu {
        let max_dim = threadgroups_per_row.max(threadgroups_per_column);
        let min_dim = threadgroups_per_row.min(threadgroups_per_column);
        let morton_dim = max_dim.next_power_of_two();
        let morton_total = morton_dim.saturating_mul(morton_dim);
        let actual_total = threadgroups_per_row.saturating_mul(threadgroups_per_column);
        let use_morton = min_dim > 1 && morton_total <= 4_u32.saturating_mul(actual_total);
        if use_morton {
            (true, morton_total, 1)
        } else {
            (false, threadgroups_per_row, threadgroups_per_column)
        }
    } else {
        (false, threadgroups_per_row, threadgroups_per_column)
    };

    let alignment = GemmAlignment::from_axes(GemmAlignmentAxes {
        m: arguments.batch_dim % tiling.block_m() == 0,
        n: arguments.output_dim % tiling.block_n() == 0,
        k: arguments.input_dim % k_block == 0,
    });
    let output_transform = output_transform_from(&arguments);

    let default_ldb = if arguments.b_transpose {
        arguments.input_dim
    } else {
        arguments.output_dim
    };
    let params = GemmParams {
        M: arguments.batch_dim,
        N: arguments.output_dim,
        K: arguments.input_dim,
        leading_dimension_a: arguments.input_dim,
        leading_dimension_b: arguments.b_leading_dimension.unwrap_or(default_ldb),
        leading_dimension_d: arguments.output_dim,
        threadgroups_per_row,
        threadgroups_per_column,
        aligned_inner_iterations: arguments.input_dim / k_block,
        use_morton,
        ab_scale: arguments.ab_scale,
    };

    let dispatch = GemmDispatch {
        tiling,
        input_prologue: GemmInputPrologueKind::FullPrecision,
        use_mxu,
        output_transform,
        alignment,
        transpose_b: arguments.b_transpose,
        a: arguments.a,
        a_offset: arguments.a_offset,
        b: GemmWeights::FullPrecision {
            weights: arguments.b,
        },
        b_offset: arguments.b_offset,
        d: arguments.d,
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

fn select_simdgroup_tiling(arguments: &MatmulArguments<Metal>) -> GemmTiling {
    if 2 * arguments.batch_dim.max(arguments.output_dim) > arguments.input_dim {
        GemmTiling::T64x64x16_2x2
    } else {
        GemmTiling::T64x32x32_2x2
    }
}

fn select_mxu_tiling(arguments: &MatmulArguments<Metal>) -> GemmTiling {
    if arguments.batch_dim >= 256 && arguments.output_dim >= 128 {
        GemmTiling::T128x128x32_4x4
    } else if arguments.output_dim < 64 {
        GemmTiling::T64x32x32_4x1
    } else if arguments.batch_dim < 64 {
        GemmTiling::T32x64x32_2x2
    } else {
        GemmTiling::T64x64x32_2x2
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
