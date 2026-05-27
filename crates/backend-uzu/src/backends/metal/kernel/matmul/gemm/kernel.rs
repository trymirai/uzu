use std::collections::{HashMap, hash_map::Entry};

use super::{MXU_THREADGROUP_BLOCK_K, specialization::GemmSpecialization};
use crate::{
    DataType,
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Backend, Buffer, Encoder,
            gpu_types::{
                GemmParams, HadamardTransformOrder,
                gemm::{GemmAlignment, GemmDTransform, GemmTiling},
            },
            kernel::{
                HadamardTransformKernel, Kernels, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulB, MatmulError, MatmulQuantCombo},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{GemmMetalKernel, TensorAddBiasMetalKernel},
            metal_extensions::DeviceExt,
        },
    },
};

#[derive(Debug, Clone, Copy)]
pub enum GemmDispatchPath {
    Simdgroup,
    Mxu,
}

pub struct GemmKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmMetalKernel>,
    pub bias_add: TensorAddBiasMetalKernel,
    pub hadamard: <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel,
}

impl GemmKernel {
    pub(crate) fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        let bias_add = TensorAddBiasMetalKernel::new(context, output_data_type, weights_data_type, true)?;
        let hadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            output_data_type,
            HadamardTransformOrder::Output,
        )?;
        let mut kernel = Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            kernels: HashMap::new(),
            bias_add,
            hadamard,
        };
        for specialization in GemmSpecialization::precompile_configs(weights_data_type) {
            kernel.get_or_create(context, specialization)?;
        }
        Ok(kernel)
    }

    pub fn preheat_quant_combo(
        &mut self,
        context: &MetalContext,
        combo: MatmulQuantCombo,
    ) -> Result<(), MetalError> {
        for specialization in GemmSpecialization::quant_combo_specs(self.weights_data_type, combo) {
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
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    specialization.use_mxu,
                    specialization.weight_prologue,
                    specialization.bits_per_weight.unwrap_or(0),
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
        let path = if encoder.context().device.supports_mxu()
            && [self.weights_data_type, self.input_data_type, self.output_data_type]
                .into_iter()
                .all(|data_type| matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32))
            && matches!(arguments.b, MatmulB::FullPrecision { .. })
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
                    [self.weights_data_type, self.input_data_type, self.output_data_type]
                        .into_iter()
                        .all(|data_type| matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)),
                    "GemmDispatchPath::Mxu requires F16, BF16, or F32 data types, got weights {:?}, input {:?}, output {:?}",
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                );
                assert!(
                    matches!(arguments.b, MatmulB::FullPrecision { .. }),
                    "GemmDispatchPath::Mxu requires FullPrecision B",
                );
                true
            },
            GemmDispatchPath::Simdgroup => false,
        };

        let is_quant = !matches!(arguments.b, MatmulB::FullPrecision { .. });
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
        let post_bias = if use_mxu {
            arguments.d_transform.bias
        } else {
            None
        };
        let output_bias = if use_mxu {
            None
        } else {
            arguments.d_transform.bias
        };
        let post_rht = arguments.d_transform.rht_factors;
        let mut output_transform = arguments.d_transform.mask();
        output_transform.remove(GemmDTransform::RHT);
        if use_mxu {
            output_transform.remove(GemmDTransform::BIAS);
        }

        let weight_prologue = arguments.b.weight_prologue();
        let bits_per_weight = arguments.b.bits_per_weight();
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
                    weights_data_type: self.weights_data_type,
                    tiling,
                    use_mxu,
                    output_transform,
                    alignment,
                    transpose_b: b_transpose,
                    weight_prologue,
                    bits_per_weight,
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
            }
            | MatmulB::ScaleSymmetricDequant {
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
                    MatmulB::ScaleSymmetricDequant {
                        b: w,
                        scales,
                        ..
                    } => (w, Some(scales), None, None),
                    _ => unreachable!(),
                };

                let tiling = select_quant_tiling(m, n, group_size.unwrap());
                let alignment =
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);
                let params = quant_params(m, n, k, tiling, ab_scale);
                let group_count_x = n.div_ceil(tiling.block_n());
                let group_count_y = m.div_ceil(tiling.block_m());

                let specialization = GemmSpecialization {
                    weights_data_type: self.weights_data_type,
                    tiling,
                    use_mxu,
                    output_transform,
                    alignment,
                    transpose_b: true,
                    weight_prologue,
                    bits_per_weight,
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
                    std::slice::from_ref(&params),
                    group_count_x,
                    group_count_y,
                    encoder,
                );
            },
        }

        if let Some(bias) = post_bias {
            self.bias_add.encode(None::<&<Metal as Backend>::DenseBuffer>, bias, &mut *d, n, m * n, encoder);
        }
        if let Some(factors) = post_rht {
            self.hadamard.encode(d, factors, n, m, encoder);
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

fn select_quant_tiling(
    m: u32,
    n: u32,
    group_size: u32,
) -> GemmTiling {
    if group_size < 32 {
        GemmTiling::T64x64x16_2x2
    } else if m < 32 {
        GemmTiling::T8x32x32_1x1
    } else if m >= 64 && n <= 2048 {
        GemmTiling::T64x32x32_2x2
    } else if m >= 64 && n >= 6144 && n % 64 == 0 {
        GemmTiling::T64x64x32_2x2
    } else {
        GemmTiling::T32x32x32_2x2
    }
}
