use std::collections::{HashMap, hash_map::Entry};

use super::{
    MXU_THREADGROUP_BLOCK_K, dispatch::GemmDispatch, request::GemmRequest, specialization::GemmSpecialization,
    weights::GemmWeights,
};
use crate::{
    DataType,
    backends::{
        common::{
            Backend, Encoder,
            gpu_types::{
                GemmParams, QuantizationMethod,
                gemm::{
                    GemmAlignment, GemmInputPrologueKind, GemmOutputTransformKind, GemmTiling, GemmWeightPrologueKind,
                },
            },
            kernel::{
                TensorAddBiasKernel,
                matmul::{MatmulArgumentC, MatmulArguments},
                quant_matmul::QuantizedMatmulConfiguration,
            },
        },
        metal::{Metal, context::MetalContext, error::MetalError, kernel::GemmMetalKernel},
    },
};

pub(crate) struct GemmKernel {
    data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmMetalKernel>,
}

impl GemmKernel {
    pub(crate) fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MetalError> {
        let mut kernel = Self {
            data_type,
            kernels: HashMap::new(),
        };
        for specialization in GemmSpecialization::precompile_configs(data_type) {
            kernel.get_or_create(context, specialization)?;
        }
        Ok(kernel)
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemmSpecialization,
    ) -> Result<&GemmMetalKernel, MetalError> {
        match self.kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmMetalKernel::new(
                    context,
                    self.data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    specialization.use_mxu,
                    specialization.weight_prologue,
                    specialization.bits_per_weight,
                    specialization.group_size,
                    specialization.input_prologue,
                    specialization.output_transform,
                    specialization.alignment,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    /// Unified entry point — encodes both FP and quantized GEMMs.
    pub(crate) fn encode(
        &mut self,
        context: &MetalContext,
        encoder: &mut Encoder<Metal>,
        request: GemmRequest<'_>,
    ) -> Result<(), MetalError> {
        match request {
            GemmRequest::Fp {
                bias_add,
                arguments,
                use_mxu,
            } => {
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

                let alignment = GemmAlignment::new(
                    arguments.batch_dim % tiling.block_m() == 0,
                    arguments.output_dim % tiling.block_n() == 0,
                    arguments.input_dim % k_block == 0,
                );
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

                self.encode_dispatch(
                    context,
                    GemmDispatch {
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
                        d: &mut *arguments.d,
                        params,
                        group_count_x,
                        group_count_y,
                    },
                    encoder,
                )?;

                if let MatmulArgumentC::Bias(bias) = arguments.c {
                    bias_add.encode(
                        None::<&<Metal as Backend>::DenseBuffer>,
                        bias,
                        arguments.d,
                        arguments.output_dim,
                        arguments.batch_dim * arguments.output_dim,
                        encoder,
                    );
                }
                Ok(())
            },
            GemmRequest::Quant {
                configuration,
                arguments,
            } => {
                let batch_dim = arguments.batch_dim as u32;
                let input_dim = configuration.input_dim as u32;
                let output_dim = configuration.output_dim as u32;
                let group_size = configuration.group_size as u32;
                let tiling = select_quant_tiling(configuration, batch_dim);
                let threadgroups_per_row = output_dim.div_ceil(tiling.block_n());
                let threadgroups_per_column = batch_dim.div_ceil(tiling.block_m());
                self.encode_dispatch(
                    context,
                    GemmDispatch {
                        tiling,
                        input_prologue: GemmInputPrologueKind::FullPrecision,
                        use_mxu: false,
                        output_transform: GemmOutputTransformKind::Store,
                        alignment: GemmAlignment::new(
                            batch_dim % tiling.block_m() == 0,
                            output_dim % tiling.block_n() == 0,
                            input_dim % tiling.block_k() == 0,
                        ),
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
                    },
                    encoder,
                )
            },
        }
    }

    fn encode_dispatch(
        &mut self,
        context: &MetalContext,
        dispatch: GemmDispatch<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let specialization = dispatch.specialization();
        assert_eq!(
            specialization.input_prologue,
            GemmInputPrologueKind::FullPrecision,
            "unified GEMM only implements FullPrecision input prologue today",
        );
        specialization.validate().map_err(MetalError::InvalidGemmSpecialization)?;
        let kernel = self.get_or_create(context, specialization)?;
        let (b, scales, biases, zero_points) = match &dispatch.b {
            GemmWeights::FullPrecision {
                weights,
            } => (*weights, None, None, None),
            GemmWeights::ScaleBias {
                weights,
                scales,
                biases,
                ..
            } => {
                debug_assert_eq!(specialization.weight_prologue, GemmWeightPrologueKind::ScaleBiasDequant);
                (*weights, Some(*scales), Some(*biases), None)
            },
            GemmWeights::ScaleZeroPoint {
                weights,
                scales,
                zero_points,
                ..
            } => {
                debug_assert_eq!(specialization.weight_prologue, GemmWeightPrologueKind::ScaleZeroPointDequant);
                (*weights, Some(*scales), None, Some(*zero_points))
            },
        };
        kernel.encode(
            (dispatch.a, dispatch.a_offset),
            (b, dispatch.b_offset),
            dispatch.d,
            scales,
            biases,
            zero_points,
            std::slice::from_ref(&dispatch.params),
            dispatch.group_count_x,
            dispatch.group_count_y,
            encoder,
        );
        Ok(())
    }
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

fn select_quant_tiling(
    configuration: &QuantizedMatmulConfiguration,
    batch_dim: u32,
) -> GemmTiling {
    if batch_dim < 32 {
        return GemmTiling::T8x32x32_1x1;
    }
    if batch_dim < 48 {
        return GemmTiling::T32x32x32_2x2;
    }

    let aligned_n_64 = configuration.output_dim % 64 == 0;
    let can_use_64_tiling = aligned_n_64 && configuration.data_type == DataType::BF16;

    if can_use_64_tiling {
        if matches!(configuration.group_size, 64 | 128) {
            GemmTiling::T64x64x64_2x2
        } else {
            GemmTiling::T64x64x32_2x2
        }
    } else {
        GemmTiling::T32x32x32_2x2
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
