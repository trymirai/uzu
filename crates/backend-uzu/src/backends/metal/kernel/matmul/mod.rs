use std::mem::size_of;

pub mod gemm;
pub mod gemv;
mod routed;

pub use self::gemm::GemmKernel;
use self::{
    gemm::{GemmPlan, GemmProblem},
    gemv::{GemvDispatch, GemvSpecialization, Mxfp4ExpertDecodeGemvDispatch, Mxfp4ExpertDecodeGemvSpec},
    routed::RoutedGemm,
};
use crate::{
    backends::{
        common::{
            BufferArg, Encoder,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform, GemmTiling},
            kernel::{
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{
                    A8ActivationPlan, ActivationFormat, GateActMulDOps, MatmulArguments, MatmulError, MatmulKernel,
                    MatmulShape, validate_matmul_storage,
                },
            },
        },
        metal::{Metal, context::MetalContext, device_profile::DeviceProfile, error::MetalError},
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvDispatch,
    routed_gemm: RoutedGemm,
    mxfp4_expert_decode: Mxfp4ExpertDecodeGemvDispatch,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    device_profile: DeviceProfile,
}

enum MatmulDispatch {
    Mxfp4ExpertDecodeGemv(Mxfp4ExpertDecodeGemvSpec),
    Gemv(GemvSpecialization),
    Gemm(GemmPlan),
}

impl MatmulMetalKernel {
    fn prefer_gemm_over_gemv(
        shape: MatmulShape,
        plan: GemmPlan,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> bool {
        if shape.sparse_readout
            || shape.expert_routed
            || shape.b_microfloat.is_some()
            || plan.engine != gemm::GemmEngine::Mxu
        {
            return false;
        }
        match (shape.m, shape.n == shape.k, (weights_data_type, input_data_type, output_data_type)) {
            (4, true, (DataType::F32, DataType::F32, DataType::F32))
            | (5, _, (DataType::BF16, DataType::BF16, DataType::BF16)) => return false,
            _ => {},
        }
        match shape.m {
            0..=3 => return false,
            4 => {
                let small_enough_for_mxu = shape.n <= 6144 && shape.k <= 9728;
                let k_dominates = shape.k > 3_u32.saturating_mul(shape.n);
                if !(small_enough_for_mxu || k_dominates) {
                    return false;
                }
            },
            _ => {},
        }
        matches!(plan.tiling, GemmTiling::Tile16x32x256_Simdgroups1x1 | GemmTiling::Tile16x128x256_Simdgroups1x4)
    }

    fn select_dispatch(
        &self,
        shape: &MatmulShape,
        ab_scale: f32,
        gate_act: Option<GateActMulDOps>,
        context: &MetalContext,
    ) -> MatmulDispatch {
        if let Some(spec) = Mxfp4ExpertDecodeGemvSpec::select(
            shape,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            ab_scale,
            gate_act,
            context.device_profile(),
        ) {
            return MatmulDispatch::Mxfp4ExpertDecodeGemv(spec);
        }
        let gemv = GemvSpecialization::select_shape(
            shape,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            context.device_profile(),
        );
        let problem = GemmProblem::new(
            *shape,
            self.weights_data_type,
            self.output_data_type,
            context.supports_mxu(),
            context.device_profile(),
        );
        let plan = problem.select_plan();
        match gemv {
            None => MatmulDispatch::Gemm(plan),
            Some(_)
                if Self::prefer_gemm_over_gemv(
                    *shape,
                    plan,
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                ) =>
            {
                MatmulDispatch::Gemm(plan)
            },
            Some(gemv) => MatmulDispatch::Gemv(gemv),
        }
    }
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
                return Err(MatmulError::<Metal>::UnsupportedDataType(data_type).into());
            }
        }

        let gemm = GemmKernel::new(context, weights_data_type, input_data_type, output_data_type)?;
        let gemv = GemvDispatch::new(weights_data_type, input_data_type, output_data_type);
        let routed_gemm = RoutedGemm::new(context, weights_data_type, input_data_type, output_data_type)?;
        let mxfp4_expert_decode =
            Mxfp4ExpertDecodeGemvDispatch::new(weights_data_type, input_data_type, output_data_type);

        Ok(Self {
            gemv,
            routed_gemm,
            mxfp4_expert_decode,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
            device_profile: context.device_profile(),
        })
    }

    fn a8_activation_plan(
        &self,
        shape: &MatmulShape,
        context: &MetalContext,
    ) -> Option<A8ActivationPlan> {
        let activation_group_size = ACTIVATION_SCALE_GROUP_SIZE;
        if !context.supports_mxu()
            || self.input_data_type != DataType::BF16
            || self.output_data_type != DataType::BF16
            || shape.a_full_precision
            || !shape.is_quant()
            || !shape.signed_codes
            || !shape.b_transpose
            || shape.b_leading_dimension.is_some()
            || !matches!(shape.b_bits, Some(4 | 8))
            || !matches!(shape.b_group_size, Some(32 | 64 | 128))
            || !shape.k.is_multiple_of(activation_group_size)
            || !shape.k.is_multiple_of(shape.b_group_size.unwrap())
        {
            return None;
        }

        let sum_group_size = match shape.b_prologue {
            GemmBPrologueKind::ScaleSymmetricDequant => None,
            GemmBPrologueKind::ScaleBiasDequant | GemmBPrologueKind::ScaleZeroPointDequant => {
                Some(shape.b_group_size.unwrap().min(activation_group_size))
            },
            GemmBPrologueKind::FullPrecision => return None,
        };
        Some(A8ActivationPlan {
            activation_group_size,
            sum_group_size,
        })
    }

    fn select_activation_format(
        &self,
        bf16_shape: &MatmulShape,
        context: &MetalContext,
    ) -> ActivationFormat {
        if matches!(
            self.select_dispatch(bf16_shape, 1.0, None, context),
            MatmulDispatch::Gemv(_) | MatmulDispatch::Mxfp4ExpertDecodeGemv(_)
        ) {
            return ActivationFormat::Bf16;
        }

        let a8_shape = MatmulShape {
            a_full_precision: false,
            ..*bf16_shape
        };
        match self.select_dispatch(&a8_shape, 1.0, None, context) {
            MatmulDispatch::Gemm(plan) if plan.engine == gemm::GemmEngine::Mxu => ActivationFormat::Int8,
            _ => ActivationFormat::Bf16,
        }
    }

    fn supports_fused_gate_act(
        &self,
        shape: &MatmulShape,
    ) -> bool {
        let probed_shape = MatmulShape {
            d_transform: GemmDTransform::GATE_ACT_MUL,
            ..*shape
        };
        Mxfp4ExpertDecodeGemvSpec::select(
            &probed_shape,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
            1.0,
            Some(GateActMulDOps {
                activation_alpha: None,
                gate_clipping: None,
                value_clipping: None,
            }),
            self.device_profile,
        )
        .is_some()
    }

    fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        if arguments.d_transform.per_matrix_bias.is_some() && arguments.routing.expert_routes().is_none() {
            return Err(MatmulError::UnsupportedRouting {
                path: "MetalMatmul",
                reason: "per-matrix bias requires direct expert routes",
            }
            .into());
        }
        if let Some(routes) = arguments.routing.expert_routes() {
            if arguments.d_transform.per_matrix_bias.is_some()
                && arguments.d_transform.mask().contains(crate::backends::common::gpu_types::gemm::GemmDTransform::RHT)
            {
                return Err(MatmulError::UnsupportedRouting {
                    path: "MetalMatmul",
                    reason: "expert bias banks cannot be combined with output RHT",
                }
                .into());
            }
            let required_ids = (arguments.m as usize).checked_mul(size_of::<i32>());
            if required_ids.is_none_or(|required| routes.expert_ids.size() < required) {
                return Err(MatmulError::UnsupportedRouting {
                    path: "MetalMatmul",
                    reason: "expert_ids must contain at least M entries",
                }
                .into());
            }
            if routes.input == crate::backends::common::kernel::matmul::ExpertInput::Tokens
                && !arguments.m.is_multiple_of(routes.routes_per_token.get())
            {
                return Err(MatmulError::UnsupportedRouting {
                    path: "MetalMatmul",
                    reason: "M must be divisible by routes_per_token for token inputs",
                }
                .into());
            }
            let required_biases = (routes.expert_count.get() as usize)
                .checked_mul(arguments.n as usize)
                .and_then(|size| size.checked_mul(self.weights_data_type.size_in_bytes()));
            if arguments
                .d_transform
                .per_matrix_bias
                .is_some_and(|biases| required_biases.is_none_or(|required| biases.size() < required))
            {
                return Err(MatmulError::UnsupportedRouting {
                    path: "MetalMatmul",
                    reason: "expert bias bank must contain expert_count * N values",
                }
                .into());
            }
            if let crate::backends::common::kernel::matmul::MatmulB::FullPrecision {
                b,
            } = &arguments.b
            {
                let leading_dimension = arguments.b_leading_dimension.unwrap_or(if arguments.b_transpose {
                    arguments.k
                } else {
                    arguments.n
                });
                let major_dimension = if arguments.b_transpose {
                    arguments.n
                } else {
                    arguments.k
                };
                let minimum_leading_dimension = if arguments.b_transpose {
                    arguments.k
                } else {
                    arguments.n
                };
                let required_bytes = (routes.expert_count.get() as usize)
                    .checked_mul(major_dimension as usize)
                    .and_then(|size| size.checked_mul(leading_dimension as usize))
                    .and_then(|size| size.checked_mul(self.weights_data_type.size_in_bytes()));
                if leading_dimension < minimum_leading_dimension
                    || required_bytes.is_none_or(|required| (*b).into_parts().2 < required)
                {
                    return Err(MatmulError::UnsupportedRouting {
                        path: "MetalMatmul",
                        reason: "full-precision weight bank layout or storage does not cover every expert matrix",
                    }
                    .into());
                }
            }
        }
        if let crate::backends::common::kernel::matmul::MatmulB::Microfloat {
            codes,
            scales,
            outer_scales,
            metadata,
        } = &arguments.b
        {
            let matrix_count = arguments.routing.expert_routes().map_or(1, |routes| routes.expert_count.get());
            let rows_match = arguments.routing.sparse_readout_rows().is_some() || metadata.rows() == arguments.n;
            if !arguments.b_transpose
                || arguments.b_leading_dimension.is_some()
                || metadata.matrix_count() < matrix_count
                || !rows_match
                || metadata.columns() != arguments.k
                || codes.size() < metadata.required_code_bytes()
                || scales.size() < metadata.required_scale_bytes()
                || (metadata.matrix_count() as usize)
                    .checked_mul(self.weights_data_type.size_in_bytes())
                    .is_none_or(|required| outer_scales.size() < required)
            {
                return Err(MatmulError::UnsupportedRouting {
                    path: "MetalMatmul",
                    reason: "microfloat storage does not match the requested matrix operand",
                }
                .into());
            }
        }
        if arguments.routing.expert_routes().is_none()
            && [self.weights_data_type, self.input_data_type, self.output_data_type].contains(&DataType::F16)
        {
            return Err(MatmulError::UnsupportedRouting {
                path: "MetalMatmul",
                reason: "F16 Metal matmul is supported only for direct expert routes",
            }
            .into());
        }
        if let Err(error) = validate_matmul_storage(&arguments, self.input_data_type, self.output_data_type) {
            return Err(MatmulError::InvalidStorage {
                path: "MetalMatmul",
                operand: error.operand,
                reason: error.reason,
            }
            .into());
        }
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = match self.select_dispatch(
            &shape,
            arguments.d_transform.ab_scale,
            arguments.d_transform.gate_act,
            encoder.context(),
        ) {
            MatmulDispatch::Mxfp4ExpertDecodeGemv(spec) => {
                return self.mxfp4_expert_decode.encode(arguments, spec, encoder).map_err(MetalError::from);
            },
            MatmulDispatch::Gemv(gemv) => {
                if arguments.d_transform.gate_act.is_some() {
                    return Err(MatmulError::<Metal>::UnsupportedDOp {
                        bit: GemmDTransform::GATE_ACT_MUL,
                        path: "MetalMatmul",
                    }
                    .into());
                }
                return self.gemv.encode(arguments, gemv, encoder).map_err(MetalError::from);
            },
            MatmulDispatch::Gemm(plan) => plan,
        };

        if arguments.d_transform.gate_act.is_some() {
            return Err(MatmulError::<Metal>::UnsupportedDOp {
                bit: GemmDTransform::GATE_ACT_MUL,
                path: "MetalMatmul",
            }
            .into());
        }

        if arguments.routing.expert_routes().is_some() || shape.b_microfloat.is_some() {
            return self.routed_gemm.encode(arguments, encoder).map_err(MetalError::from);
        }
        // TODO: remove after GatherGEMM is supported
        if arguments.routing.sparse_readout_rows().is_some() {
            return Err(MetalError::KernelDispatchFailed(
                format!(
                    "gathered readout requires the GEMV path, but shape (m={}, n={}) routes to GEMM",
                    arguments.m, arguments.n
                )
                .into(),
            ));
        }
        self.gemm.encode_plan(arguments, plan, encoder)
    }
}
