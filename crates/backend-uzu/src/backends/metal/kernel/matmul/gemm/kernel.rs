use std::collections::{HashMap, hash_map::Entry};

use super::{MXU_THREADGROUP_BLOCK_K, specialization::GemmSpecialization};
use crate::{
    DataType,
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Buffer, Encoder,
            gpu_types::{
                GemmParams,
                gemm::{GemmAlignment, GemmDTransform, GemmTiling},
            },
            kernel::{
                gemm::GemmWeights,
                matmul::{MatmulArguments, MatmulB, MatmulDOp},
            },
        },
        metal::{
            Metal, context::MetalContext, error::MetalError, kernel::GemmMetalKernel, metal_extensions::DeviceExt,
        },
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
                    specialization.data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    specialization.use_mxu,
                    specialization.weight_prologue,
                    specialization.bits_per_weight,
                    specialization.group_size,
                    specialization.output_transform,
                    specialization.alignment,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        context: &MetalContext,
        arguments: MatmulArguments<'a, Metal, TB>,
        force_simdgroup: bool,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let use_mxu = !force_simdgroup
            && context.device.supports_mxu()
            && matches!(self.data_type, DataType::F16 | DataType::BF16)
            && matches!(arguments.b, MatmulB::FullPrecision { .. });

        let ab_scale = arguments.d_transform.iter().find_map(|op| op.as_scale()).unwrap_or(1.0);
        let output_bias = arguments.d_transform.iter().find_map(|op| op.as_bias());
        let output_transform = MatmulDOp::mask(&arguments.d_transform) - GemmDTransform::RHT;

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
                    ab_scale,
                };

                let weights_gw: GemmWeights<'_, Metal, TB> = GemmWeights::FullPrecision {
                    weights,
                };
                let specialization = GemmSpecialization {
                    data_type: self.data_type,
                    tiling,
                    use_mxu,
                    output_transform,
                    alignment,
                    transpose_b: b_transpose,
                    weight_prologue: weights_gw.weight_prologue(),
                    bits_per_weight: weights_gw.bits_per_weight(),
                    group_size: weights_gw.group_size(),
                };
                specialization.validate().map_err(MetalError::from)?;
                let kernel = self.get_or_create(context, specialization)?;
                kernel.encode(
                    (a, a_offset),
                    (weights, b_offset),
                    d,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    output_bias,
                    std::slice::from_ref(&params),
                    group_count_x,
                    group_count_y,
                    encoder,
                );
            },
            quant_b @ (MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }) => {
                let (weights, weights_gw, scales, biases, zero_points) = match quant_b {
                    MatmulB::ScaleBiasDequant {
                        b: w,
                        scales,
                        biases,
                        mode,
                        group_size,
                    } => (
                        w,
                        GemmWeights::ScaleBias {
                            weights: w,
                            scales,
                            biases,
                            mode,
                            group_size,
                        },
                        Some(scales),
                        Some(biases),
                        None,
                    ),
                    MatmulB::ScaleZeroPointDequant {
                        b: w,
                        scales,
                        zero_points,
                        mode,
                        group_size,
                    } => (
                        w,
                        GemmWeights::ScaleZeroPoint {
                            weights: w,
                            scales,
                            zero_points,
                            mode,
                            group_size,
                        },
                        Some(scales),
                        None,
                        Some(zero_points),
                    ),
                    _ => unreachable!(),
                };
                let weights_gw: GemmWeights<'_, Metal, TB> = weights_gw;

                let tiling = select_quant_tiling(m);
                let alignment =
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);
                let params = quant_params(m, n, k, tiling, ab_scale);
                let group_count_x = n.div_ceil(tiling.block_n());
                let group_count_y = m.div_ceil(tiling.block_m());

                let specialization = GemmSpecialization {
                    data_type: self.data_type,
                    tiling,
                    use_mxu,
                    output_transform,
                    alignment,
                    transpose_b: true,
                    weight_prologue: weights_gw.weight_prologue(),
                    bits_per_weight: weights_gw.bits_per_weight(),
                    group_size: weights_gw.group_size(),
                };
                specialization.validate().map_err(MetalError::from)?;
                let kernel = self.get_or_create(context, specialization)?;
                kernel.encode(
                    (a, a_offset),
                    (weights, b_offset),
                    d,
                    scales,
                    biases,
                    zero_points,
                    output_bias,
                    std::slice::from_ref(&params),
                    group_count_x,
                    group_count_y,
                    encoder,
                );
            },
        }
        Ok(())
    }
}

fn quant_params(
    m: u32,
    n: u32,
    k: u32,
    tiling: GemmTiling,
    ab_scale: f32,
) -> GemmParams {
    GemmParams {
        M: m,
        N: n,
        K: k,
        leading_dimension_a: k,
        leading_dimension_b: k,
        leading_dimension_d: n,
        threadgroups_per_row: n.div_ceil(tiling.block_n()),
        threadgroups_per_column: m.div_ceil(tiling.block_m()),
        aligned_inner_iterations: k / tiling.block_k(),
        use_morton: false,
        ab_scale,
    }
}

fn select_simdgroup_tiling(
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

fn select_mxu_tiling(
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

fn select_quant_tiling(m: u32) -> GemmTiling {
    if m < 32 {
        GemmTiling::T8x32x32_1x1
    } else {
        GemmTiling::T32x32x32_2x2
    }
}
