use std::collections::{HashMap, hash_map::Entry};

use super::{
    super::supports_integer_right_operand,
    GemmEngine, GemmPlan,
    selection::{GemmProblem, outer_block_k},
    specialization::GemmSpecialization,
};
use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            gpu_types::{
                GemmParams,
                gemm::{GemmAPrologueKind, GemmAlignment, GemmBPrologueKind, GemmDTransform},
            },
            kernel::{
                ActivationTransform, TensorAddBiasKernel,
                matmul::{
                    Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape, QuantParamsLayout,
                    QuantParamsStrides,
                },
            },
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
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
        context: &MetalContext,
    ) -> GemmProblem {
        GemmProblem::new(
            shape,
            self.weights_data_type,
            self.output_data_type,
            context.supports_mxu,
            context.apple_gpu_family,
        )
    }

    #[cfg(test)]
    fn select_plan_for_engine(
        &self,
        shape: &MatmulShape,
        engine: GemmEngine,
        context: &MetalContext,
    ) -> Result<GemmPlan, MetalError> {
        self.problem(*shape, context)
            .select_plan_for_engine(engine)
            .map_err(|error| MetalError::KernelDispatchFailed(Box::new(error)))
    }

    #[cfg(test)]
    pub fn encode_with_engine(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        engine: GemmEngine,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = self.select_plan_for_engine(&shape, engine, command_buffer.context())?;
        self.encode_plan(arguments, plan, command_buffer)
    }

    pub fn encode_plan(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        plan: GemmPlan,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        self.problem(shape, command_buffer.context())
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

        let MatmulArguments {
            a,
            b,
            b_leading_dimension,
            b_transpose,
            mut d,
            m,
            n,
            k,
            ..
        } = arguments;

        let use_mxu = plan.engine == GemmEngine::Mxu;

        let quantized = match b {
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
                        None::<&<Metal as Backend>::GlobalBuffer>,
                        None::<&<Metal as Backend>::GlobalBuffer>,
                        None::<&<Metal as Backend>::GlobalBuffer>,
                        d,
                        ab_scale,
                        None,
                        None,
                        shape,
                        plan,
                        output_transform,
                        output_bias,
                        rht_factors,
                        command_buffer,
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
                    scale_output_stride: 0,
                    scale_group_stride: 0,
                    zero_point_output_stride: 0,
                    zero_point_group_stride: 0,
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
                let kernel = self.get_or_create(command_buffer.context(), specialization)?;
                kernel.encode(
                    Some(a.subrange(a_offset..)),
                    weights,
                    d,
                    None::<&<Metal as Backend>::GlobalBuffer>,
                    None::<&<Metal as Backend>::GlobalBuffer>,
                    None::<&<Metal as Backend>::GlobalBuffer>,
                    output_bias,
                    rht_factors,
                    None::<&<Metal as Backend>::ScratchBuffer>,
                    None::<&<Metal as Backend>::ScratchBuffer>,
                    None::<&<Metal as Backend>::ScratchBuffer>,
                    std::slice::from_ref(&params),
                    group_count_x,
                    group_count_y,
                    1,
                    command_buffer,
                );
                return Ok(());
            },
            MatmulB::Quantized(quantized) => quantized,
        };
        let a_prologue = a.prologue_kind();
        let (a_full_precision, a_int8, a_scales, a_group_sums, a_group_size) = match &a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => (Some(values.subrange(*offset..)), None, None, None, None),
            MatmulA::Int8Symmetric {
                values,
                scales: activation_scales,
                group_sums: activation_group_sums,
                scale_group_size,
                code_layout,
            } => {
                validate_int8_left_operand(
                    use_mxu,
                    shape,
                    *scale_group_size,
                    *code_layout,
                    activation_group_sums.is_some(),
                )?;
                if output_transform.contains(GemmDTransform::SOFT_CAP) {
                    return Err(MatmulError::UnsupportedDOp {
                        bit: GemmDTransform::SOFT_CAP,
                        path: "Gemm int8 left operand",
                    }
                    .into());
                }
                (None, Some(*values), Some(*activation_scales), *activation_group_sums, Some(*scale_group_size))
            },
        };

        let (output_bias, bias_after_rht, output_transform) = if rht_factors.is_some() && output_bias.is_some() {
            (None, output_bias, output_transform.difference(GemmDTransform::BIAS))
        } else {
            (output_bias, None, output_transform)
        };

        let tiling = plan.tiling;
        let alignment =
            GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);
        let scale_strides = quantized.params.scale_strides();
        let zero_point_strides = quantized.zero_point_strides();
        let params = gemm_params(shape, plan, ab_scale, scale_strides, zero_point_strides);
        let group_count_x = n.div_ceil(tiling.block_n());
        let group_count_y = m.div_ceil(tiling.block_m());

        if plan.split_k > 1 {
            self.encode_split_k(
                a,
                quantized.codes,
                Some(quantized.scales),
                quantized.biases(),
                quantized.zero_points(),
                d.reborrow(),
                ab_scale,
                Some(scale_strides),
                Some(zero_point_strides),
                shape,
                plan,
                output_transform,
                output_bias,
                rht_factors,
                command_buffer,
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
            let kernel = self.get_or_create(command_buffer.context(), specialization)?;
            kernel.encode(
                a_full_precision,
                quantized.codes,
                d.reborrow(),
                Some(quantized.scales),
                quantized.biases(),
                quantized.zero_points(),
                output_bias,
                rht_factors,
                a_int8,
                a_scales,
                a_group_sums,
                std::slice::from_ref(&params),
                group_count_x,
                group_count_y,
                1,
                command_buffer,
            );
        }

        if let Some(bias) = bias_after_rht {
            let output_length = m.checked_mul(n).expect("GEMM output length must fit in u32");
            self.bias_add.encode(None::<&<Metal as Backend>::ScratchBuffer>, bias, d, n, output_length, command_buffer);
        }

        Ok(())
    }

    fn encode_split_k(
        &mut self,
        a: MatmulA<impl BufferRef<Backend = Metal>>,
        weights: impl BufferRef<Backend = Metal>,
        scales: Option<impl BufferRef<Backend = Metal>>,
        biases: Option<impl BufferRef<Backend = Metal>>,
        zero_points: Option<impl BufferRef<Backend = Metal>>,
        mut d: impl BufferMut<Backend = Metal>,
        ab_scale: f32,
        scale_strides: Option<QuantParamsStrides>,
        zero_point_strides: Option<QuantParamsStrides>,
        shape: MatmulShape,
        plan: GemmPlan,
        output_transform: GemmDTransform,
        output_bias: Option<impl BufferRef<Backend = Metal>>,
        rht_factors: Option<impl BufferRef<Backend = Metal>>,
        command_buffer: &mut MetalCommandBufferEncoding,
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
            } => (Some(values.subrange(offset..)), None, None, None, GemmAPrologueKind::FullPrecision, None),
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums,
                scale_group_size,
                code_layout: _,
            } => {
                (None, Some(values), Some(scales), group_sums, GemmAPrologueKind::Int8Symmetric, Some(scale_group_size))
            },
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
        let mut temp = command_buffer.allocate_scratch(split_k as usize * slice_bytes)?;
        let scale_strides = scale_strides.unwrap_or_default();
        let zero_point_strides = zero_point_strides.unwrap_or_default();

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
            scale_output_stride: scale_strides.output_stride,
            scale_group_stride: scale_strides.group_stride,
            zero_point_output_stride: zero_point_strides.output_stride,
            zero_point_group_stride: zero_point_strides.group_stride,
        };
        let part_kernel = self.get_or_create(command_buffer.context(), part_spec)?;
        part_kernel.encode(
            a_full_precision,
            weights,
            &mut temp,
            scales,
            biases,
            zero_points,
            None::<&<Metal as Backend>::GlobalBuffer>,
            None::<&<Metal as Backend>::GlobalBuffer>,
            a_int8,
            a_scales,
            a_group_sums,
            std::slice::from_ref(&params),
            base_gx,
            base_gy,
            split_k,
            command_buffer,
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
        let reduce = self.get_or_create_split_k_reduce(command_buffer.context(), reduce_transform)?;
        reduce.encode(&temp, d.reborrow(), bias_arg, elem as u32, split_k, group_count, n, scale_arg, command_buffer);

        if output_transform.contains(GemmDTransform::RHT)
            && let Some(factors) = rht_factors
        {
            self.output_rht.encode_fp_in_place(d, factors, m, n, command_buffer);
        }
        Ok(())
    }
}

fn validate_int8_left_operand(
    use_mxu: bool,
    shape: MatmulShape,
    a_group_size: u32,
    code_layout: Int8CodeLayout,
    has_group_sums: bool,
) -> Result<(), MetalError> {
    if shape.b_bits.and_then(Int8CodeLayout::for_right_bits) != Some(code_layout) {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "left code layout is incompatible with the right operand",
        }
        .into());
    }
    let needs_group_sums =
        matches!(shape.b_prologue, GemmBPrologueKind::ScaleBiasDequant | GemmBPrologueKind::ScaleZeroPointDequant);
    if needs_group_sums && !has_group_sums {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "quantized correction requires left group sums",
        }
        .into());
    }
    let compatible = use_mxu
        && supports_integer_right_operand(&shape)
        && shape.params_layout == Some(QuantParamsLayout::GroupOutput)
        && matches!(
            shape.b_prologue,
            GemmBPrologueKind::ScaleSymmetricDequant
                | GemmBPrologueKind::ScaleBiasDequant
                | GemmBPrologueKind::ScaleZeroPointDequant
        )
        && matches!(a_group_size, 32 | 64 | 128)
        && shape.k.is_multiple_of(a_group_size)
        && shape
            .b_group_size
            .is_some_and(|gs| matches!(gs, 32 | 64 | 128) && shape.k.is_multiple_of(gs) && a_group_size >= gs);
    if !compatible {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "symmetric int8 left operands require group-major metadata and supported 32/64/128 groups",
        }
        .into());
    }
    Ok(())
}

fn gemm_params(
    shape: MatmulShape,
    plan: GemmPlan,
    ab_scale: f32,
    scale_strides: QuantParamsStrides,
    zero_point_strides: QuantParamsStrides,
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
        scale_output_stride: scale_strides.output_stride,
        scale_group_stride: scale_strides.group_stride,
        zero_point_output_stride: zero_point_strides.output_stride,
        zero_point_group_stride: zero_point_strides.group_stride,
    }
}
