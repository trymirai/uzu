use std::mem::size_of;

pub mod gemm;
pub mod gemv;
mod routed;

pub use self::gemm::GemmKernel;
use self::{
    gemm::{GemmPlan, GemmProblem},
    gemv::{GemvDispatch, GemvSpecialization},
    routed::RoutedGemm,
};
use crate::{
    backends::{
        common::{
            BufferArg, Encoder,
            gpu_types::gemm::{GemmBPrologueKind, GemmTiling},
            kernel::{
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{A8ActivationPlan, ActivationFormat, MatmulArguments, MatmulError, MatmulKernel, MatmulShape},
            },
        },
        metal::{Metal, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvDispatch,
    routed_gemm: RoutedGemm,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

enum MatmulDispatch {
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
        if shape.gathered || shape.expert_routed || plan.engine != gemm::GemmEngine::Mxu {
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
        context: &MetalContext,
    ) -> MatmulDispatch {
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
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(MatmulError::<Metal>::UnsupportedDataType(data_type).into());
            }
        }

        let gemm = GemmKernel::new(context, weights_data_type, input_data_type, output_data_type)?;
        let gemv = GemvDispatch::new(weights_data_type, input_data_type, output_data_type);
        let routed_gemm = RoutedGemm::new(context, weights_data_type, input_data_type, output_data_type)?;

        Ok(Self {
            gemv,
            routed_gemm,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
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
        if matches!(self.select_dispatch(bf16_shape, context), MatmulDispatch::Gemv(_)) {
            return ActivationFormat::Bf16;
        }

        let a8_shape = MatmulShape {
            a_full_precision: false,
            ..*bf16_shape
        };
        match self.select_dispatch(&a8_shape, context) {
            MatmulDispatch::Gemm(plan) if plan.engine == gemm::GemmEngine::Mxu => ActivationFormat::Int8,
            _ => ActivationFormat::Bf16,
        }
    }

    fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        if arguments.gather_indices.is_some() && arguments.expert_routes.is_some() {
            return Err(MatmulError::UnsupportedRouting {
                path: "MetalMatmul",
                reason: "sparse readout and expert routing cannot be combined",
            }
            .into());
        }
        if let Some(routes) = arguments.expert_routes {
            if routes.expert_ids.size() < arguments.m as usize * size_of::<i32>() {
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
            if routes.expert_biases.is_some_and(|biases| {
                biases.size()
                    < routes.expert_count.get() as usize * arguments.n as usize * self.weights_data_type.size_in_bytes()
            }) {
                return Err(MatmulError::UnsupportedRouting {
                    path: "MetalMatmul",
                    reason: "expert bias bank must contain expert_count * N values",
                }
                .into());
            }
        }
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = match self.select_dispatch(&shape, encoder.context()) {
            MatmulDispatch::Gemv(gemv) => {
                return self.gemv.encode(arguments, gemv, encoder).map_err(MetalError::from);
            },
            MatmulDispatch::Gemm(plan) => plan,
        };

        if arguments.expert_routes.is_some() {
            return self.routed_gemm.encode(arguments, encoder).map_err(MetalError::from);
        }
        // TODO: remove after GatherGEMM is supported
        if arguments.gather_indices.is_some() {
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
