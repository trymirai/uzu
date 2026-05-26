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
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError, MatmulQuantCombo},
        },
        metal::{
            Metal, context::MetalContext, error::MetalError, kernel::GemmMetalKernel, metal_extensions::DeviceExt,
        },
    },
};

#[derive(Debug, Clone, Copy)]
pub enum GemmDispatchPath {
    Simdgroup,
    Mxu,
}

pub struct GemmKernel {
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

    pub fn preheat_quant_combo(
        &mut self,
        context: &MetalContext,
        combo: MatmulQuantCombo,
    ) -> Result<(), MetalError> {
        for specialization in GemmSpecialization::quant_combo_specs(self.data_type, combo) {
            self.get_or_create(context, specialization)?;
        }
        Ok(())
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
                    specialization.b_prologue,
                    specialization.bits_per_b.unwrap_or(0),
                    specialization.group_size.unwrap_or(0),
                    specialization.output_transform,
                    specialization.alignment,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub fn encode<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let mxu_eligible_for_quant = match &arguments.b {
            MatmulB::FullPrecision { .. } => true,
            MatmulB::ScaleBiasDequant { .. } | MatmulB::ScaleZeroPointDequant { .. } => {
                arguments.b_transpose
                    && arguments.b_leading_dimension.is_none_or(|ld| ld == arguments.k)
                    && arguments.b_offset == 0
                    && arguments.k % super::MXU_THREADGROUP_BLOCK_K == 0
            },
        };
        let path = if encoder.context().device.supports_mxu()
            && matches!(self.data_type, DataType::F16 | DataType::BF16)
            && mxu_eligible_for_quant
        {
            GemmDispatchPath::Mxu
        } else {
            GemmDispatchPath::Simdgroup
        };
        self.encode_dispatch_path(arguments, path, encoder)
    }

    pub fn encode_dispatch_path<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        path: GemmDispatchPath,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let use_mxu = match path {
            GemmDispatchPath::Mxu => {
                assert!(
                    encoder.context().device.supports_mxu(),
                    "GemmDispatchPath::Mxu requested on hardware without MXU support",
                );
                assert!(
                    matches!(self.data_type, DataType::F16 | DataType::BF16),
                    "GemmDispatchPath::Mxu requires F16 or BF16 data type, got {:?}",
                    self.data_type,
                );
                true
            },
            GemmDispatchPath::Simdgroup => false,
        };

        let is_quant = matches!(arguments.b, MatmulB::ScaleBiasDequant { .. } | MatmulB::ScaleZeroPointDequant { .. });
        if is_quant {
            let d_mask = arguments.d_transform.mask();
            if d_mask.contains(GemmDTransform::ACCUMULATE) {
                return Err(MatmulError::UnsupportedDOp {
                    bit: GemmDTransform::ACCUMULATE,
                    path: "QuantGemm",
                }
                .into());
            }
            assert!(
                !d_mask.contains(GemmDTransform::BIAS | GemmDTransform::RHT),
                "QuantGemm with both output bias and output RHT is not supported: bias must be applied after RHT",
            );
            if !arguments.b_transpose || arguments.b_leading_dimension.is_some() || arguments.b_offset != 0 {
                return Err(MatmulError::UnsupportedLayout {
                    path: "QuantGemm",
                }
                .into());
            }
        }

        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let output_transform = arguments.d_transform.mask();

        let b_prologue = arguments.b.b_prologue();
        let bits_per_b = arguments.b.bits_per_b();
        let group_size = arguments.b.group_size();

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

                let specialization = GemmSpecialization {
                    data_type: self.data_type,
                    tiling,
                    use_mxu,
                    output_transform,
                    alignment,
                    transpose_b: b_transpose,
                    b_prologue,
                    bits_per_b,
                    group_size,
                };
                specialization.validate()?;
                let kernel = self.get_or_create(encoder.context(), specialization)?;
                kernel.encode(
                    (a, a_offset),
                    (weights, b_offset),
                    &mut *d,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    output_bias,
                    rht_factors,
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
                let (weights, scales, biases, zero_points) = match quant_b {
                    MatmulB::ScaleBiasDequant {
                        b: w,
                        scales,
                        biases,
                        ..
                    } => (w, Some(scales), Some(biases), None),
                    MatmulB::ScaleZeroPointDequant {
                        b: w,
                        scales,
                        zero_points,
                        ..
                    } => (w, Some(scales), None, Some(zero_points)),
                    _ => unreachable!(),
                };

                let tiling = if use_mxu {
                    select_mxu_tiling(m, n)
                } else {
                    select_quant_tiling(m, n)
                };
                let k_block = if use_mxu {
                    MXU_THREADGROUP_BLOCK_K
                } else {
                    tiling.block_k()
                };
                let alignment =
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % k_block == 0);
                let params = quant_params(m, n, k, tiling, k_block, ab_scale);
                let group_count_x = n.div_ceil(tiling.block_n());
                let group_count_y = m.div_ceil(tiling.block_m());

                let specialization = GemmSpecialization {
                    data_type: self.data_type,
                    tiling,
                    use_mxu,
                    output_transform,
                    alignment,
                    transpose_b: true,
                    b_prologue,
                    bits_per_b,
                    group_size,
                };
                specialization.validate()?;
                let kernel = self.get_or_create(encoder.context(), specialization)?;
                kernel.encode(
                    (a, a_offset),
                    (weights, b_offset),
                    &mut *d,
                    scales,
                    biases,
                    zero_points,
                    output_bias,
                    rht_factors,
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
    k_block: u32,
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
        aligned_inner_iterations: k / k_block,
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

fn select_quant_tiling(
    m: u32,
    n: u32,
) -> GemmTiling {
    if m < 32 {
        GemmTiling::T8x32x32_1x1
    } else if m >= 64 && n <= 2048 {
        GemmTiling::T64x32x32_2x2
    } else if m >= 64 && n >= 6144 && n % 64 == 0 {
        GemmTiling::T64x64x32_2x2
    } else {
        GemmTiling::T32x32x32_2x2
    }
}
