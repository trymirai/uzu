use std::collections::{HashMap, hash_map::Entry};

use super::{
    GemmEngine, GemmPlan,
    selection::{GemmProblem, outer_block_k},
    specialization::GemmSpecialization,
};
use crate::{
    backends::{
        common::{
            Allocation, BufferArg, Encoder,
            gpu_types::{
                GemmParams,
                gemm::{GemmAPrologueKind, GemmAlignment, GemmBPrologueKind, GemmDTransform},
            },
            kernel::{
                ActivationTransform, TensorAddBiasKernel,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            device_profile::DeviceProfile,
            error::MetalError,
            kernel::{GemmMetalKernel, GemmSplitKReduceMetalKernel, TensorAddBiasMetalKernel},
        },
    },
    data_type::DataType,
};

pub struct GemmKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmMetalKernel>,
    pub bias_add: TensorAddBiasMetalKernel,
    output_rht: ActivationTransform<Metal>,
    split_k_reduce: HashMap<GemmDTransform, GemmSplitKReduceMetalKernel>,
}

impl GemmKernel {
    pub fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        let bias_add = TensorAddBiasMetalKernel::new(context, output_data_type, weights_data_type, true)?;
        let output_rht = ActivationTransform::output_rht(context, output_data_type, true)?;
        let kernel = Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            kernels: HashMap::new(),
            bias_add,
            output_rht,
            split_k_reduce: HashMap::new(),
        };
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
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    specialization.use_mxu,
                    specialization.b_prologue,
                    specialization.bits_per_b.unwrap_or(0),
                    specialization.b_group_size.unwrap_or(0),
                    specialization.a_prologue,
                    specialization.a_group_size.unwrap_or(0),
                    specialization.output_transform,
                    specialization.alignment,
                    specialization.signed_codes,
                    specialization.stage_weight_scales,
                    specialization.hoist_operand_addressing,
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

    fn problem(
        &self,
        shape: MatmulShape,
        supports_mxu: bool,
        profile: DeviceProfile,
    ) -> GemmProblem {
        GemmProblem::new(shape, self.weights_data_type, self.output_data_type, supports_mxu, profile)
    }

    #[cfg(test)]
    fn select_plan_for_engine(
        &self,
        shape: &MatmulShape,
        engine: GemmEngine,
        context: &MetalContext,
    ) -> Result<GemmPlan, MetalError> {
        self.problem(*shape, context.supports_mxu(), context.device_profile())
            .select_plan_for_engine(engine)
            .map_err(|error| MetalError::KernelDispatchFailed(Box::new(error)))
    }

    #[cfg(test)]
    pub fn encode_with_engine<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        engine: GemmEngine,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = self.select_plan_for_engine(&shape, engine, encoder.context())?;
        self.encode_plan(arguments, plan, encoder)
    }

    pub fn encode_plan<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        plan: GemmPlan,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        self.problem(shape, encoder.context().supports_mxu(), encoder.context().device_profile())
            .validate_engine(plan.engine)
            .map_err(|error| MetalError::KernelDispatchFailed(Box::new(error)))?;

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
        }

        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let output_transform = arguments.d_transform.mask();

        let b_prologue = arguments.b.b_prologue();
        let bits_per_b = arguments.b.bits_per_b();
        let group_size = arguments.b.group_size();
        let weights_signed_codes = arguments.b.signed_codes();

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

        let use_mxu = plan.engine == GemmEngine::Mxu;

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

                let tiling = plan.tiling;

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

                if plan.split_k > 1 {
                    return self.encode_split_k(
                        MatmulA::FullPrecision {
                            values: a,
                            offset: a_offset,
                        },
                        weights,
                        None,
                        None,
                        None,
                        &mut *d,
                        ab_scale,
                        shape,
                        plan,
                        output_transform,
                        output_bias,
                        rht_factors,
                        encoder,
                    );
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

                let specialization = GemmSpecialization::from_plan(
                    plan,
                    shape,
                    self.weights_data_type,
                    output_transform,
                    alignment,
                    GemmAPrologueKind::FullPrecision,
                    None,
                )?;
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
            MatmulB::Microfloat {
                ..
            } => {
                return Err(MatmulError::UnsupportedRouting {
                    path: "Gemm",
                    reason: "microfloat Metal execution is not implemented",
                }
                .into());
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
                let (a_full_precision, a_int8, a_scales, a_group_sums, a_group_size) = match &a {
                    MatmulA::FullPrecision {
                        values,
                        offset,
                    } => (Some((*values, *offset)), None, None, None, None),
                    MatmulA::Int8Symmetric {
                        values,
                        scales: activation_scales,
                        group_sums: activation_group_sums,
                        group_size: a_group_size,
                    } => {
                        validate_int8_activation_arguments(
                            use_mxu,
                            weights_signed_codes,
                            k,
                            b_prologue,
                            bits_per_b,
                            group_size,
                            *a_group_size,
                        )?;
                        if output_transform.contains(GemmDTransform::SOFT_CAP) {
                            return Err(MatmulError::UnsupportedDOp {
                                bit: GemmDTransform::SOFT_CAP,
                                path: "Gemm int8 activations",
                            }
                            .into());
                        }
                        (None, Some(*values), Some(*activation_scales), *activation_group_sums, Some(*a_group_size))
                    },
                };

                let (output_bias, bias_after_rht, output_transform) = if rht_factors.is_some() && output_bias.is_some()
                {
                    (None, output_bias, output_transform.difference(GemmDTransform::BIAS))
                } else {
                    (output_bias, None, output_transform)
                };

                let tiling = plan.tiling;
                let alignment =
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);
                let params = quant_params(shape, plan, ab_scale);
                let group_count_x = n.div_ceil(tiling.block_n());
                let group_count_y = m.div_ceil(tiling.block_m());

                if plan.split_k > 1 {
                    self.encode_split_k(
                        a,
                        weights,
                        scales,
                        biases,
                        zero_points,
                        &mut *d,
                        ab_scale,
                        shape,
                        plan,
                        output_transform,
                        output_bias,
                        rht_factors,
                        encoder,
                    )?;
                } else {
                    let specialization = GemmSpecialization::from_plan(
                        plan,
                        shape,
                        self.weights_data_type,
                        output_transform,
                        alignment,
                        a_prologue,
                        a_group_size,
                    )?;
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
                    self.bias_add.encode(None::<&Allocation<Metal>>, bias, &mut *d, n, output_length, encoder);
                }
            },
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_split_k<'a, 'b, WB: BufferArg<'b, Metal>>(
        &mut self,
        a: MatmulA<'a, Metal>,
        weights: WB,
        scales: Option<&Allocation<Metal>>,
        biases: Option<&Allocation<Metal>>,
        zero_points: Option<&Allocation<Metal>>,
        d: &mut Allocation<Metal>,
        ab_scale: f32,
        shape: MatmulShape,
        plan: GemmPlan,
        output_transform: GemmDTransform,
        output_bias: Option<&Allocation<Metal>>,
        rht_factors: Option<&Allocation<Metal>>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let MatmulShape {
            m,
            n,
            k,
            ..
        } = shape;
        let (a_full_precision, a_int8, a_scales, a_group_sums, a_prologue, a_group_size) = match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => (Some((values, offset)), None, None, None, GemmAPrologueKind::FullPrecision, None),
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums,
                group_size,
            } => (None, Some(values), Some(scales), group_sums, GemmAPrologueKind::Int8Symmetric, Some(group_size)),
        };
        let tiling = plan.tiling;
        let split_k = plan.split_k;
        let kp = k / split_k;
        let k_step = outer_block_k(shape, plan.engine, plan.tiling).unwrap_or(1);
        let base_gx = n.div_ceil(tiling.block_n());
        let base_gy = m.div_ceil(tiling.block_m());
        let alignment =
            GemmAlignment::new(m.is_multiple_of(tiling.block_m()), n.is_multiple_of(tiling.block_n()), true);
        let part_spec = GemmSpecialization::from_plan(
            plan,
            shape,
            self.weights_data_type,
            GemmDTransform::empty(),
            alignment,
            a_prologue,
            a_group_size,
        )?;

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
            self.output_rht.encode_fp_in_place(&mut *d, factors, m, n, encoder);
        }
        Ok(())
    }
}

fn validate_int8_activation_arguments(
    use_mxu: bool,
    weights_signed_codes: bool,
    k: u32,
    b_prologue: GemmBPrologueKind,
    bits_per_b: Option<u32>,
    weight_group_size: Option<u32>,
    a_group_size: u32,
) -> Result<(), MetalError> {
    let compatible = use_mxu
        && weights_signed_codes
        && matches!(
            b_prologue,
            GemmBPrologueKind::ScaleSymmetricDequant
                | GemmBPrologueKind::ScaleBiasDequant
                | GemmBPrologueKind::ScaleZeroPointDequant
        )
        && matches!(bits_per_b, Some(4 | 8))
        && matches!(a_group_size, 32 | 64 | 128)
        && k.is_multiple_of(a_group_size)
        && weight_group_size.is_some_and(|gs| matches!(gs, 32 | 64 | 128) && k.is_multiple_of(gs));
    if !compatible {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "symmetric int8 activations require a supported 32/64/128 activation and weight group",
        }
        .into());
    }
    Ok(())
}

fn quant_params(
    shape: MatmulShape,
    plan: GemmPlan,
    ab_scale: f32,
) -> GemmParams {
    let MatmulShape {
        m,
        n,
        k,
        ..
    } = shape;
    let tiling = plan.tiling;
    GemmParams {
        M: m,
        N: n,
        K: k,
        leading_dimension_a: k,
        leading_dimension_b: k,
        leading_dimension_d: n,
        threadgroups_per_row: n.div_ceil(tiling.block_n()),
        threadgroups_per_column: m.div_ceil(tiling.block_m()),
        aligned_inner_iterations: outer_block_k(shape, plan.engine, plan.tiling).map_or(0, |step| k / step),
        use_morton: false,
        ab_scale,
    }
}
