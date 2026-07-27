use std::collections::{HashMap, hash_map::Entry};

use super::specialization::GemmSpecialization;
use crate::{
    backends::{
        common::{
            Allocation, Backend, BufferArg, Encoder,
            gpu_types::{
                GemmParams, HADAMARD_TRANSFORM_BLOCK_SIZE, HadamardTransformOrder, QuantizationMode,
                gemm::{GemmAPrologueKind, GemmAlignment, GemmBPrologueKind, GemmDTransform, GemmTiling},
            },
            kernel::{
                HadamardTransformKernel, Kernels, TensorAddBiasKernel,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{GemmMetalKernel, GemmSplitKReduceMetalKernel, TensorAddBiasMetalKernel},
            metal_extensions::DeviceExt,
        },
    },
    data_type::DataType,
};

/// Measured on M5 Max over m 16..2048 x {k2048 n3072, k3584 n1024} x {sym, zp}, each
/// candidate scored against the default dispatch in the same process. 256 beats 128 in 21
/// of 32 cells by 8-17% and regresses only one cell (+3%). 512 is better still at m=16 on
/// k2048 n3072 but worse on k3584 n1024, so it is shape-dependent and not worth branching
/// on. Above m~1024 the tile count already exceeds any target and split-k stays at 1.
const SPLIT_K_TARGET_TILES_INT8_ACTIVATIONS: u32 = 256;
const SPLIT_K_TARGET_TILES: u32 = 512;

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
    split_k_reduce: HashMap<GemmDTransform, GemmSplitKReduceMetalKernel>,
    /// Test-only knobs so tuning sweeps can pin what the dispatch heuristics would pick.
    /// Absent from release builds, so the production path is exactly as before.
    #[cfg(test)]
    pub(crate) tiling_override: Option<GemmTiling>,
    #[cfg(test)]
    pub(crate) split_k_target_override: Option<u32>,
    #[cfg(test)]
    pub(crate) fused_a8_epilogue_override: Option<bool>,
}

impl GemmKernel {
    pub(crate) fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        let bias_add = TensorAddBiasMetalKernel::new(context, output_data_type, weights_data_type, true, false)?;
        let hadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            output_data_type,
            HadamardTransformOrder::Output,
        )?;
        let kernel = Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            kernels: HashMap::new(),
            bias_add,
            hadamard,
            split_k_reduce: HashMap::new(),
            #[cfg(test)]
            tiling_override: None,
            #[cfg(test)]
            split_k_target_override: None,
            #[cfg(test)]
            fused_a8_epilogue_override: None,
        };
        Ok(kernel)
    }

    fn use_fused_a8_epilogue(
        &self,
        m: u32,
        a_prologue: GemmAPrologueKind,
        group_size: Option<u32>,
    ) -> bool {
        let selected =
            m <= 32 && a_prologue == GemmAPrologueKind::Int8Symmetric && group_size.is_some_and(|size| size >= 64);
        #[cfg(test)]
        let selected = self.fused_a8_epilogue_override.unwrap_or(selected);
        selected
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
                    specialization.b_prologue,
                    specialization.bits_per_b.unwrap_or(0),
                    specialization.group_size.unwrap_or(0),
                    specialization.a_prologue,
                    specialization.signed_w8_storage,
                    specialization.fused_a8_epilogue,
                    specialization.output_transform,
                    specialization.alignment,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_split_k_reduce(
        &mut self,
        context: &MetalContext,
        output_transform: GemmDTransform,
    ) -> Result<&GemmSplitKReduceMetalKernel, MetalError> {
        match self.split_k_reduce.entry(output_transform) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmSplitKReduceMetalKernel::new(context, self.output_data_type, output_transform)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn should_skip_gemv_for_mxu<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &self,
        arguments: &MatmulArguments<'a, 'b, 'd, Metal, TB>,
    ) -> bool {
        if arguments.gather_indices.is_some() {
            // TODO: gathered GEMM
            return false;
        }
        match (
            arguments.m,
            arguments.n == arguments.k,
            (self.weights_data_type, self.input_data_type, self.output_data_type),
        ) {
            (4, true, (DataType::F32, DataType::F32, DataType::F32))
            | (5, _, (DataType::BF16, DataType::BF16, DataType::BF16)) => return false,
            _ => {},
        }
        match arguments.m {
            0..=3 => return false,
            4 => {
                // The M4 MXU tile only uses a quarter of its rows; avoid it for wide-N shapes.
                let small_enough_for_mxu = arguments.n <= 6144 && arguments.k <= 9728;
                let k_dominates = arguments.k > 3_u32.saturating_mul(arguments.n);
                if !(small_enough_for_mxu || k_dominates) {
                    return false;
                }
            },
            _ => {},
        }
        matches!(
            self.select_mxu_tiling(arguments),
            Some(GemmTiling::Tile16x32x256_Simdgroups1x1 | GemmTiling::Tile16x128x256_Simdgroups1x4)
        )
    }

    fn select_mxu_tiling<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &self,
        arguments: &MatmulArguments<'a, 'b, 'd, Metal, TB>,
    ) -> Option<GemmTiling> {
        if ![self.weights_data_type, self.input_data_type, self.output_data_type]
            .into_iter()
            .all(|data_type| matches!(data_type, DataType::BF16 | DataType::F32))
        {
            return None;
        }

        match &arguments.b {
            MatmulB::FullPrecision {
                ..
            } => Some(if arguments.b_transpose {
                select_mxu_tiling(arguments.m, arguments.n, arguments.k)
            } else {
                select_base_mxu_tiling(arguments.m, arguments.n)
            }),
            MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                ..
            } => {
                if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
                    return None;
                }
                let group_size = arguments.b.group_size().unwrap_or(0);
                let int8_activations = arguments.a.prologue_kind() == GemmAPrologueKind::Int8Symmetric;
                let tiling = select_mxu_quant_tiling(arguments.m, arguments.n, group_size, int8_activations);
                if int8_activations {
                    return (group_size != 0 && arguments.k.is_multiple_of(group_size)).then_some(tiling);
                }
                arguments.k.is_multiple_of(tiling.block_k()).then_some(tiling)
            },
        }
    }

    pub fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let path = if encoder.context().device.supports_mxu()
            && (arguments.a.prologue_kind() == GemmAPrologueKind::Int8Symmetric
                || self.select_mxu_tiling(&arguments).is_some())
        {
            GemmDispatchPath::Mxu
        } else {
            GemmDispatchPath::Simdgroup
        };
        self.encode_dispatch_path(arguments, path, encoder)
    }

    pub fn encode_dispatch_path<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        path: GemmDispatchPath,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        if matches!(path, GemmDispatchPath::Mxu) {
            assert!(
                encoder.context().device.supports_mxu(),
                "GemmDispatchPath::Mxu requested on hardware without MXU support",
            );
            assert!(
                [self.weights_data_type, self.input_data_type, self.output_data_type]
                    .into_iter()
                    .all(|data_type| matches!(data_type, DataType::BF16 | DataType::F32)),
                "GemmDispatchPath::Mxu requires BF16 or F32 data types, got weights {:?}, input {:?}, output {:?}",
                self.weights_data_type,
                self.input_data_type,
                self.output_data_type,
            );
        }

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
            if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
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
        let signed_w8_storage = arguments.b.quantization_mode() == Some(QuantizationMode::I8);

        let MatmulArguments {
            a,
            b,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        let use_mxu = matches!(path, GemmDispatchPath::Mxu);

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                let MatmulA::FullPrecision {
                    values: a,
                    offset: a_offset,
                } = a
                else {
                    return Err(MatmulError::IncompatibleA {
                        path: "Gemm",
                        reason: "int8 activations require quantized weights",
                    }
                    .into());
                };

                let tiling = if use_mxu {
                    if b_transpose {
                        select_mxu_tiling(m, n, k)
                    } else {
                        select_base_mxu_tiling(m, n)
                    }
                } else {
                    select_simdgroup_tiling(m, n, k)
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
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);

                if b_transpose && b_leading_dimension.is_none() {
                    let split_k = select_split_k(m, n, k, tiling, use_mxu, 0, true, false, SPLIT_K_TARGET_TILES);
                    if split_k > 1
                        && split_k_output_supported(output_transform, n, self.weights_data_type, self.output_data_type)
                    {
                        return self.encode_split_k(
                            Some((a, a_offset)),
                            None,
                            None,
                            None,
                            GemmAPrologueKind::FullPrecision,
                            weights,
                            None,
                            None,
                            None,
                            &mut *d,
                            m,
                            n,
                            k,
                            ab_scale,
                            use_mxu,
                            tiling,
                            b_prologue,
                            bits_per_b,
                            group_size,
                            false,
                            split_k,
                            output_transform,
                            output_bias,
                            rht_factors,
                            encoder,
                        );
                    }
                }

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
                    aligned_inner_iterations: k / tiling.block_k(),
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
                    b_prologue,
                    bits_per_b,
                    group_size,
                    a_prologue: GemmAPrologueKind::FullPrecision,
                    signed_w8_storage: false,
                    fused_a8_epilogue: false,
                };
                specialization.validate()?;
                let kernel = self.get_or_create(encoder.context(), specialization)?;
                kernel.encode(
                    Some((a, a_offset)),
                    weights,
                    &mut *d,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    output_bias,
                    rht_factors,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    std::slice::from_ref(&params),
                    group_count_x,
                    group_count_y,
                    1,
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

                let a_prologue = a.prologue_kind();
                let a_is_int8 = a_prologue == GemmAPrologueKind::Int8Symmetric;
                let (a_full_precision, a_int8, a_scales, a_group_sums) = match a {
                    MatmulA::FullPrecision {
                        values,
                        offset,
                    } => (Some((values, offset)), None, None, None),
                    MatmulA::Int8Symmetric {
                        values,
                        scales: activation_scales,
                        group_sums: activation_group_sums,
                    } => {
                        validate_int8_activation_arguments(use_mxu, k, b_prologue, bits_per_b, group_size)?;
                        if output_transform.contains(GemmDTransform::SOFT_CAP) {
                            return Err(MatmulError::UnsupportedDOp {
                                bit: GemmDTransform::SOFT_CAP,
                                path: "Gemm int8 activations",
                            }
                            .into());
                        }
                        (None, Some(values), Some(activation_scales), activation_group_sums)
                    },
                };

                let (output_bias, bias_after_rht, output_transform) = if rht_factors.is_some() && output_bias.is_some()
                {
                    (None, output_bias, output_transform.difference(GemmDTransform::BIAS))
                } else {
                    (output_bias, None, output_transform)
                };

                let tiling = {
                    let selected = if use_mxu {
                        select_mxu_quant_tiling(m, n, group_size.unwrap_or(0), a_is_int8)
                    } else {
                        select_quant_tiling(m, n, group_size.unwrap_or(0))
                    };
                    #[cfg(test)]
                    let selected = self.tiling_override.unwrap_or(selected);
                    selected
                };
                let alignment =
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);
                let params = quant_params(m, n, k, tiling, use_mxu, group_size.unwrap_or(0), ab_scale);
                let group_count_x = n.div_ceil(tiling.block_n());
                let group_count_y = m.div_ceil(tiling.block_m());

                let zero_point_4bit = zero_points.is_some() && bits_per_b == Some(4);
                let split_k_target = if a_is_int8 {
                    SPLIT_K_TARGET_TILES_INT8_ACTIVATIONS
                } else {
                    SPLIT_K_TARGET_TILES
                };
                #[cfg(test)]
                let split_k_target = if a_is_int8 {
                    self.split_k_target_override.unwrap_or(split_k_target)
                } else {
                    split_k_target
                };
                let split_k = select_split_k(
                    m,
                    n,
                    k,
                    tiling,
                    use_mxu,
                    group_size.unwrap_or(0),
                    false,
                    zero_point_4bit,
                    split_k_target,
                );
                if split_k > 1
                    && split_k_output_supported(output_transform, n, self.weights_data_type, self.output_data_type)
                {
                    self.encode_split_k(
                        a_full_precision,
                        a_int8,
                        a_scales,
                        a_group_sums,
                        a_prologue,
                        weights,
                        scales,
                        biases,
                        zero_points,
                        &mut *d,
                        m,
                        n,
                        k,
                        ab_scale,
                        use_mxu,
                        tiling,
                        b_prologue,
                        bits_per_b,
                        group_size,
                        signed_w8_storage,
                        split_k,
                        output_transform,
                        output_bias,
                        rht_factors,
                        encoder,
                    )?;
                } else {
                    let specialization = GemmSpecialization {
                        weights_data_type: self.weights_data_type,
                        tiling,
                        use_mxu,
                        output_transform,
                        alignment,
                        transpose_b: true,
                        b_prologue,
                        bits_per_b,
                        group_size,
                        a_prologue,
                        signed_w8_storage,
                        fused_a8_epilogue: self.use_fused_a8_epilogue(m, a_prologue, group_size),
                    };
                    specialization.validate()?;
                    let kernel = self.get_or_create(encoder.context(), specialization)?;
                    kernel.encode(
                        a_full_precision,
                        weights,
                        &mut *d,
                        scales,
                        biases,
                        zero_points,
                        output_bias,
                        rht_factors,
                        a_int8,
                        a_scales,
                        a_group_sums,
                        std::slice::from_ref(&params),
                        group_count_x,
                        group_count_y,
                        1,
                        encoder,
                    );
                }

                if let Some(bias) = bias_after_rht {
                    let output_length = m.checked_mul(n).expect("GEMM output length must fit in u32");
                    self.bias_add.encode(
                        None::<&Allocation<Metal>>,
                        bias,
                        None::<&Allocation<Metal>>,
                        &mut *d,
                        n,
                        output_length,
                        encoder,
                    );
                }
            },
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_split_k<'a, 'b, WB: BufferArg<'b, Metal>>(
        &mut self,
        a_full_precision: Option<(&Allocation<Metal>, usize)>,
        a_int8: Option<&Allocation<Metal>>,
        a_scales: Option<&Allocation<Metal>>,
        a_group_sums: Option<&Allocation<Metal>>,
        a_prologue: GemmAPrologueKind,
        weights: WB,
        scales: Option<&Allocation<Metal>>,
        biases: Option<&Allocation<Metal>>,
        zero_points: Option<&Allocation<Metal>>,
        d: &mut Allocation<Metal>,
        m: u32,
        n: u32,
        k: u32,
        ab_scale: f32,
        use_mxu: bool,
        tiling: GemmTiling,
        b_prologue: GemmBPrologueKind,
        bits_per_b: Option<u32>,
        group_size: Option<u32>,
        signed_w8_storage: bool,
        split_k: u32,
        output_transform: GemmDTransform,
        output_bias: Option<&Allocation<Metal>>,
        rht_factors: Option<&Allocation<Metal>>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let full_precision = matches!(b_prologue, GemmBPrologueKind::FullPrecision);
        let kp = k / split_k;
        let k_step = split_k_step(tiling, use_mxu, group_size.unwrap_or(0), full_precision).unwrap_or(1);
        let base_gx = n.div_ceil(tiling.block_n());
        let base_gy = m.div_ceil(tiling.block_m());
        let alignment =
            GemmAlignment::new(m.is_multiple_of(tiling.block_m()), n.is_multiple_of(tiling.block_n()), true);
        let part_spec = GemmSpecialization {
            weights_data_type: self.weights_data_type,
            tiling,
            use_mxu,
            output_transform: GemmDTransform::empty(),
            alignment,
            transpose_b: true,
            b_prologue,
            bits_per_b,
            group_size,
            a_prologue,
            signed_w8_storage,
            fused_a8_epilogue: self.use_fused_a8_epilogue(m, a_prologue, group_size),
        };
        part_spec.validate()?;

        let elem = (m as usize) * (n as usize);
        let slice_bytes = elem * self.output_data_type.size_in_bytes();
        let mut temp = encoder.allocate_scratch(split_k as usize * slice_bytes)?;

        let params = GemmParams {
            M: m,
            N: n,
            K: k,
            leading_dimension_a: k,
            leading_dimension_b: k,
            leading_dimension_d: n,
            threadgroups_per_row: base_gx,
            threadgroups_per_column: base_gy,
            aligned_inner_iterations: kp / k_step,
            use_morton: false,
            ab_scale: 1.0,
        };
        let part_kernel = self.get_or_create(encoder.context(), part_spec)?;
        part_kernel.encode(
            a_full_precision,
            weights,
            &mut temp,
            scales,
            biases,
            zero_points,
            None::<&Allocation<Metal>>,
            None::<&Allocation<Metal>>,
            a_int8,
            a_scales,
            a_group_sums,
            std::slice::from_ref(&params),
            base_gx,
            base_gy,
            split_k,
            encoder,
        );

        debug_assert_eq!(elem % 4, 0, "split-K reduce requires M*N divisible by 4");
        let group_count = ((elem as u32) / 4).div_ceil(256);
        let reduce_transform =
            output_transform.intersection(GemmDTransform::SCALE | GemmDTransform::ACCUMULATE | GemmDTransform::BIAS);
        let bias_arg = if reduce_transform.contains(GemmDTransform::BIAS) {
            output_bias
        } else {
            None
        };
        let scale_arg = if reduce_transform.contains(GemmDTransform::SCALE) {
            Some(ab_scale)
        } else {
            None
        };
        let reduce = self.get_or_create_split_k_reduce(encoder.context(), reduce_transform)?;
        reduce.encode((&temp, 0usize), &mut *d, bias_arg, elem as u32, split_k, group_count, n, scale_arg, encoder);

        if output_transform.contains(GemmDTransform::RHT)
            && let Some(factors) = rht_factors
        {
            self.hadamard.encode(&mut *d, factors, n, m, encoder);
        }
        Ok(())
    }
}

fn validate_int8_activation_arguments(
    use_mxu: bool,
    k: u32,
    b_prologue: GemmBPrologueKind,
    bits_per_b: Option<u32>,
    weight_group_size: Option<u32>,
) -> Result<(), MetalError> {
    let weight_gs_ok = matches!(weight_group_size, Some(32 | 64 | 128));
    let compatible = use_mxu
        && matches!(
            b_prologue,
            GemmBPrologueKind::ScaleSymmetricDequant
                | GemmBPrologueKind::ScaleBiasDequant
                | GemmBPrologueKind::ScaleZeroPointDequant
        )
        && matches!(bits_per_b, Some(4 | 8))
        && weight_gs_ok
        && (k as usize).is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)
        && weight_group_size.is_some_and(|gs| k.is_multiple_of(gs));
    if !compatible {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "symmetric int8 activations require MXU and 4/8-bit quantized weights with group size 32/64/128",
        }
        .into());
    }
    Ok(())
}

fn quant_params(
    m: u32,
    n: u32,
    k: u32,
    tiling: GemmTiling,
    use_mxu: bool,
    group_size: u32,
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
        aligned_inner_iterations: split_k_step(tiling, use_mxu, group_size, false).map_or(0, |step| k / step),
        use_morton: false,
        ab_scale,
    }
}

fn split_k_step(
    tiling: GemmTiling,
    use_mxu: bool,
    group_size: u32,
    full_precision: bool,
) -> Option<u32> {
    let step = if use_mxu && !full_precision {
        group_size
    } else {
        tiling.block_k()
    };
    (step != 0).then_some(step)
}

fn split_k_output_supported(
    output_transform: GemmDTransform,
    n: u32,
    weights_data_type: DataType,
    output_data_type: DataType,
) -> bool {
    if !output_transform.contains(GemmDTransform::BIAS) {
        return true;
    }
    n.is_multiple_of(4) && weights_data_type == output_data_type
}

pub(crate) fn select_split_k(
    m: u32,
    n: u32,
    k: u32,
    tiling: GemmTiling,
    use_mxu: bool,
    group_size: u32,
    full_precision: bool,
    zero_point_4bit: bool,
    target_tiles: u32,
) -> u32 {
    let base_tiles = n.div_ceil(tiling.block_n()) * m.div_ceil(tiling.block_m());
    if base_tiles == 0 {
        return 1;
    }
    if !((m as u64) * (n as u64)).is_multiple_of(4) {
        return 1;
    }
    let mut split_k = (target_tiles / base_tiles).max(1);
    let step = match split_k_step(tiling, use_mxu, group_size, full_precision) {
        Some(s) => s,
        None => return 1,
    };
    let mut align = if use_mxu || full_precision {
        step
    } else {
        step.max(group_size)
    };
    if zero_point_4bit {
        align = align.max(2 * group_size);
    }
    split_k = split_k.min((k / align).max(1));
    while split_k > 1 && !k.is_multiple_of(split_k * align) {
        split_k -= 1;
    }
    split_k
}

pub(crate) fn select_simdgroup_tiling(
    m: u32,
    n: u32,
    k: u32,
) -> GemmTiling {
    if 2 * m.max(n) > k {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else {
        GemmTiling::Tile64x32x32_Simdgroups2x2
    }
}

pub(crate) fn select_mxu_tiling(
    m: u32,
    n: u32,
    k: u32,
) -> GemmTiling {
    if m < 64 && n >= 64 {
        if n == k {
            return if m < 16 && k <= 2560 {
                GemmTiling::Tile16x32x256_Simdgroups1x1
            } else {
                GemmTiling::Tile32x64x256_Simdgroups2x2
            };
        }
        return if m < 16 {
            select_small_m_mxu_tiling(n, k)
        } else {
            select_base_mxu_tiling(m, n)
        };
    }
    select_base_mxu_tiling(m, n)
}

fn select_base_mxu_tiling(
    m: u32,
    n: u32,
) -> GemmTiling {
    if m >= 256 && n >= 128 {
        GemmTiling::Tile128x128x256_Simdgroups4x4
    } else if n < 64 {
        GemmTiling::Tile64x32x256_Simdgroups4x1
    } else if m < 64 {
        GemmTiling::Tile32x64x256_Simdgroups2x2
    } else {
        GemmTiling::Tile64x64x256_Simdgroups2x2
    }
}

fn select_small_m_mxu_tiling(
    n: u32,
    k: u32,
) -> GemmTiling {
    if k > n {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    if n > 32_u32.saturating_mul(k) {
        return GemmTiling::Tile16x32x256_Simdgroups1x1;
    }
    if (k >= 4096 && n >= 4_u32.saturating_mul(k)) || (k == 2560 && n >= 6_u32.saturating_mul(k)) {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    GemmTiling::Tile32x64x256_Simdgroups2x2
}

/// int8 activations want a different tile table from bf16 activations. The a8 path stages
/// no weights through threadgroup memory, so it is register- rather than LDS-limited, and
/// the bf16 table mispicks at both ends of the m range.
///
/// Measured on M5 Max, w8, group size 64, m 16..2048 x 5 shapes x {sym, zp}, every
/// candidate measured back to back with the incumbent in one process and scored against a
/// bracketing pair of incumbent readings.
fn select_int8_mxu_tiling(
    m: u32,
    n: u32,
) -> GemmTiling {
    if n < 64 {
        return GemmTiling::Tile64x32x256_Simdgroups4x1;
    }
    // m == 16 fell through to a block_m=32 tile, wasting half the m dimension: a block_m=16
    // tile is 15-43% faster on every shape measured. Which of the two 16-row tilings wins
    // does not follow n, k or n/k, but this one has the better mean (0.68 sym / 0.64 zp
    // against the incumbent, versus 0.71 / 0.69 for Tile16x32).
    if m <= 16 {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    if m < 64 {
        return GemmTiling::Tile32x64x256_Simdgroups2x2;
    }
    // The bf16 table promotes to 128x128 at m >= 256. For int8 that is premature: at m=256
    // Tile64x64 averages 0.941 over all shape/method cells (worst +3.2%, best -17.1%),
    // while at m >= 1024 the promotion is correct (Tile64x64 averages 1.019 there). The two
    // share a per-thread register footprint, so the larger tile buys no extra reuse - only
    // coarser m granularity, which is what costs at 256.
    if m >= 512 {
        return GemmTiling::Tile128x128x256_Simdgroups4x4;
    }
    GemmTiling::Tile64x64x256_Simdgroups2x2
}

pub(crate) fn select_mxu_quant_tiling(
    m: u32,
    n: u32,
    group_size: u32,
    int8_activations: bool,
) -> GemmTiling {
    let tiling = if int8_activations {
        select_int8_mxu_tiling(m, n)
    } else {
        select_base_mxu_tiling(m, n)
    };
    if tiling.fits_quant_group_size(group_size) {
        tiling
    } else {
        GemmTiling::Tile64x64x256_Simdgroups2x2
    }
}

pub(crate) fn select_quant_tiling(
    m: u32,
    n: u32,
    group_size: u32,
) -> GemmTiling {
    if group_size < 32 {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else if m < 32 {
        GemmTiling::Tile8x32x32_Simdgroups1x1
    } else if m >= 64 && n <= 2048 {
        GemmTiling::Tile32x32x32_Simdgroups2x2
    } else if m >= 64 && n >= 6144 && n.is_multiple_of(64) {
        GemmTiling::Tile64x64x32_Simdgroups2x2
    } else {
        GemmTiling::Tile32x32x32_Simdgroups2x2
    }
}
