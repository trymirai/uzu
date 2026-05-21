use std::collections::{HashMap, hash_map::Entry};

use super::{
    MXU_THREADGROUP_BLOCK_K, dispatch::GemmDispatch, specialization::GemmSpecialization, weights::GemmWeights,
};
use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::{
                GemmParams, QuantizationMethod,
                gemm::{GemmAlignment, GemmInputPrologueKind, GemmTiling, GemmWeightPrologueKind},
            },
            kernel::matmul::{MatmulArguments, MatmulB, MatmulDOp},
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

    /// Unified entry point — encodes both FP and quantized GEMMs. Match arm on
    /// `args.b` selects the path; `use_mxu` is honored only by the FP path.
    pub(crate) fn encode<'a>(
        &mut self,
        context: &MetalContext,
        encoder: &mut Encoder<Metal>,
        arguments: MatmulArguments<'a, Metal>,
        use_mxu: bool,
    ) -> Result<(), MetalError> {
        // DSL: read scale/bias state directly from d_transform.
        let ab_scale = arguments.d_transform.iter().find_map(|op| op.as_scale()).unwrap_or(1.0);
        let output_bias = arguments.d_transform.iter().find_map(|op| op.as_bias());
        let core_kind = MatmulDOp::mask(&arguments.d_transform)
            .core_kind()
            .expect("unsupported D-transform combination for unified GEMM");

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                let tiling = if use_mxu {
                    select_mxu_tiling(m, n)
                } else {
                    select_simdgroup_tiling(m, n, k)
                };
                let k_block = if use_mxu {
                    MXU_THREADGROUP_BLOCK_K
                } else {
                    tiling.block_k()
                };

                let threadgroups_per_row = n.div_ceil(tiling.block_n());
                let threadgroups_per_column = m.div_ceil(tiling.block_m());

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

                let alignment =
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % k_block == 0);
                let output_transform = core_kind;

                let default_ldb = if b_transpose {
                    k
                } else {
                    n
                };
                let params = GemmParams {
                    M: m,
                    N: n,
                    K: k,
                    leading_dimension_a: k,
                    leading_dimension_b: b_leading_dimension.unwrap_or(default_ldb),
                    leading_dimension_d: n,
                    threadgroups_per_row,
                    threadgroups_per_column,
                    aligned_inner_iterations: k / k_block,
                    use_morton,
                    ab_scale: ab_scale,
                };

                self.encode_dispatch(
                    context,
                    GemmDispatch {
                        tiling,
                        input_prologue: GemmInputPrologueKind::FullPrecision,
                        use_mxu,
                        output_transform,
                        alignment,
                        transpose_b: b_transpose,
                        a,
                        a_offset,
                        b: GemmWeights::FullPrecision {
                            weights,
                        },
                        b_offset,
                        d,
                        output_bias,
                        params,
                        group_count_x,
                        group_count_y,
                    },
                    encoder,
                )
            },
            MatmulB::ScaleBiasDequant {
                b: weights,
                scales,
                biases,
                mode,
                group_size,
            } => self.encode_quant(
                context,
                encoder,
                weights,
                scales,
                biases,
                QuantizationMethod::ScaleBias,
                mode.into(),
                group_size,
                a,
                a_offset,
                b_offset,
                d,
                output_bias,
                core_kind,
                ab_scale,
                m,
                n,
                k,
            ),
            MatmulB::ScaleZeroPointDequant {
                b: weights,
                scales,
                zero_points,
                mode,
                group_size,
            } => self.encode_quant(
                context,
                encoder,
                weights,
                scales,
                zero_points,
                QuantizationMethod::ScaleZeroPoint,
                mode.into(),
                group_size,
                a,
                a_offset,
                b_offset,
                d,
                output_bias,
                core_kind,
                ab_scale,
                m,
                n,
                k,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_quant<'a>(
        &mut self,
        context: &MetalContext,
        encoder: &mut Encoder<Metal>,
        weights: &'a crate::backends::common::Allocation<Metal>,
        scales: &'a crate::backends::common::Allocation<Metal>,
        zero_points_or_biases: &'a crate::backends::common::Allocation<Metal>,
        method: QuantizationMethod,
        mode: crate::backends::common::gpu_types::QuantizationMode,
        group_size: u32,
        a: &'a crate::backends::common::Allocation<Metal>,
        a_offset: usize,
        b_offset: usize,
        d: &'a mut crate::backends::common::Allocation<Metal>,
        output_bias: Option<&'a crate::backends::common::Allocation<Metal>>,
        output_transform: crate::backends::common::gpu_types::gemm::GemmOutputTransformKind,
        ab_scale: f32,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), MetalError> {
        let tiling = select_quant_tiling(self.data_type, m, n, group_size);
        let threadgroups_per_row = n.div_ceil(tiling.block_n());
        let threadgroups_per_column = m.div_ceil(tiling.block_m());
        self.encode_dispatch(
            context,
            GemmDispatch {
                tiling,
                input_prologue: GemmInputPrologueKind::FullPrecision,
                use_mxu: false,
                output_transform,
                alignment: GemmAlignment::new(
                    m % tiling.block_m() == 0,
                    n % tiling.block_n() == 0,
                    k % tiling.block_k() == 0,
                ),
                transpose_b: true,
                a,
                a_offset,
                b: match method {
                    QuantizationMethod::ScaleBias => GemmWeights::ScaleBias {
                        weights,
                        scales,
                        biases: zero_points_or_biases,
                        mode,
                        group_size,
                    },
                    QuantizationMethod::ScaleZeroPoint => GemmWeights::ScaleZeroPoint {
                        weights,
                        scales,
                        zero_points: zero_points_or_biases,
                        mode,
                        group_size,
                    },
                },
                b_offset,
                d,
                output_bias,
                params: GemmParams {
                    M: m,
                    N: n,
                    K: k,
                    leading_dimension_a: k,
                    leading_dimension_b: k,
                    leading_dimension_d: n,
                    threadgroups_per_row,
                    threadgroups_per_column,
                    aligned_inner_iterations: k / tiling.block_k(),
                    use_morton: false,
                    ab_scale,
                },
                group_count_x: threadgroups_per_row,
                group_count_y: threadgroups_per_column,
            },
            encoder,
        )
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
            dispatch.output_bias,
            std::slice::from_ref(&dispatch.params),
            dispatch.group_count_x,
            dispatch.group_count_y,
            encoder,
        );
        Ok(())
    }
}

pub(crate) fn select_simdgroup_tiling(
    m: u32,
    n: u32,
    k: u32,
) -> GemmTiling {
    if 2 * m.max(n) > k {
        GemmTiling::T64x64x16_2x2
    } else {
        GemmTiling::T64x32x32_2x2
    }
}

pub(crate) fn select_mxu_tiling(
    m: u32,
    n: u32,
) -> GemmTiling {
    if m >= 256 && n >= 128 {
        GemmTiling::T128x128x32_4x4
    } else if n < 64 {
        GemmTiling::T64x32x32_4x1
    } else if m < 64 {
        GemmTiling::T32x64x32_2x2
    } else {
        GemmTiling::T64x64x32_2x2
    }
}

pub(crate) fn select_quant_tiling(
    data_type: DataType,
    m: u32,
    n: u32,
    group_size: u32,
) -> GemmTiling {
    if m < 32 {
        return GemmTiling::T8x32x32_1x1;
    }
    if m < 48 {
        return GemmTiling::T32x32x32_2x2;
    }

    let aligned_n_64 = n % 64 == 0;
    let can_use_64_tiling = aligned_n_64 && data_type == DataType::BF16;

    if can_use_64_tiling {
        if matches!(group_size, 64 | 128) {
            GemmTiling::T64x64x64_2x2
        } else {
            GemmTiling::T64x64x32_2x2
        }
    } else {
        GemmTiling::T32x32x32_2x2
    }
}
