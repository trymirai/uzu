use super::{MXU_THREADGROUP_BLOCK_K, specialization::GemmSpecialization, weights::GemmWeights};
use crate::{
    DataType,
    backends::{
        common::{
            Allocation,
            gpu_types::{
                GemmParams,
                gemm::{GemmAlignment, GemmDTransform, GemmTiling},
            },
            kernel::matmul::{MatmulArguments, MatmulB, MatmulDOp},
        },
        metal::Metal,
    },
};

pub struct GemmDispatch<'a> {
    pub tiling: GemmTiling,
    pub use_mxu: bool,
    pub output_transform: GemmDTransform,
    pub alignment: GemmAlignment,
    pub transpose_b: bool,
    pub a: &'a Allocation<Metal>,
    pub a_offset: usize,
    pub b: GemmWeights<'a, Metal>,
    pub b_offset: usize,
    pub d: &'a mut Allocation<Metal>,
    pub output_bias: Option<&'a Allocation<Metal>>,
    pub params: GemmParams,
    pub group_count_x: u32,
    pub group_count_y: u32,
}

impl<'a> GemmDispatch<'a> {
    /// Build a dispatch from a public-API `MatmulArguments`. Picks the tiling
    /// (FP simdgroup / FP MXU / quant simdgroup), decides Morton ordering for
    /// MXU, computes alignment + params, and selects the `GemmWeights` variant
    /// from `MatmulB`.
    pub(crate) fn from_arguments(
        arguments: MatmulArguments<'a, Metal>,
        data_type: DataType,
        mxu_eligible: bool,
    ) -> Self {
        // DSL: read scale/bias state directly from d_transform. The bitmask IS
        // the wire format; strip post-pass-only bits (RHT) since the kernel
        // doesn't see those.
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
                let use_mxu = mxu_eligible;
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

                let alignment = GemmAlignment::new(
                    m % tiling.block_m() == 0,
                    n % tiling.block_n() == 0,
                    k % k_block == 0,
                );

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

                Self {
                    tiling,
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
                }
            },
            MatmulB::ScaleBiasDequant {
                b: weights,
                scales,
                biases,
                mode,
                group_size,
            } => {
                let tiling = select_quant_tiling(data_type, m, n, group_size);
                let (group_count_x, group_count_y) = (n.div_ceil(tiling.block_n()), m.div_ceil(tiling.block_m()));
                Self {
                    tiling,
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
                    b: GemmWeights::ScaleBias {
                        weights,
                        scales,
                        biases,
                        mode,
                        group_size,
                    },
                    b_offset,
                    d,
                    output_bias,
                    params: quant_params(m, n, k, tiling, ab_scale),
                    group_count_x,
                    group_count_y,
                }
            },
            MatmulB::ScaleZeroPointDequant {
                b: weights,
                scales,
                zero_points,
                mode,
                group_size,
            } => {
                let tiling = select_quant_tiling(data_type, m, n, group_size);
                let (group_count_x, group_count_y) = (n.div_ceil(tiling.block_n()), m.div_ceil(tiling.block_m()));
                Self {
                    tiling,
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
                    b: GemmWeights::ScaleZeroPoint {
                        weights,
                        scales,
                        zero_points,
                        mode,
                        group_size,
                    },
                    b_offset,
                    d,
                    output_bias,
                    params: quant_params(m, n, k, tiling, ab_scale),
                    group_count_x,
                    group_count_y,
                }
            },
        }
    }

    pub(crate) fn specialization(&self) -> GemmSpecialization {
        GemmSpecialization {
            tiling: self.tiling,
            use_mxu: self.use_mxu,
            output_transform: self.output_transform,
            alignment: self.alignment,
            transpose_b: self.transpose_b,
            weight_prologue: self.b.weight_prologue(),
            bits_per_weight: self.b.bits_per_weight(),
            group_size: self.b.group_size(),
        }
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
