pub mod gemm;
mod gemv;
mod qmv;

use metal::MTLGPUFamily;

pub use self::gemm::GemmKernel;
use self::{
    gemm::{GemmPlan, GemmProblem},
    gemv::{GemvKernel, GemvSpecialization},
    qmv::QmvRoute,
};
use crate::{
    backends::{
        common::{
            BufferMut, BufferRef, CommandBufferEncoding,
            gpu_types::gemm::{GemmBPrologueKind, GemmTiling},
            kernel::{
                ActivationQuantization,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{
                    ActivationFormat, Int8CodeLayout, MatmulArguments, MatmulError, MatmulKernel, MatmulShape,
                    QuantParamsLayout,
                },
            },
        },
        metal::{Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvKernel,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

enum MatmulDispatch {
    Gemv(GemvSpecialization),
    Gemm(GemmPlan),
}

fn supports_integer_right_operand(shape: &MatmulShape) -> bool {
    matches!((shape.b_bits, shape.signed_codes), (Some(4), _) | (Some(8), true))
}

impl MatmulMetalKernel {
    fn prefer_gemm_over_gemv(
        shape: MatmulShape,
        plan: GemmPlan,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> bool {
        if shape.gathered || plan.engine != gemm::GemmEngine::Mxu {
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

    fn choose_dispatch(
        shape: &MatmulShape,
        device_name: &str,
        gpu_core_count: u32,
        apple_gpu_family: MTLGPUFamily,
        supports_mxu: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> MatmulDispatch {
        let all_bf16 = weights_data_type == DataType::BF16
            && input_data_type == DataType::BF16
            && output_data_type == DataType::BF16;
        if let Some(route) = qmv::route(device_name, apple_gpu_family, supports_mxu, shape, all_bf16) {
            return match route {
                QmvRoute::Tuned(tile) | QmvRoute::MainGemv(tile) => MatmulDispatch::Gemv(
                    GemvSpecialization::select_tile(shape, weights_data_type, input_data_type, output_data_type, tile)
                        .expect("typed QMV route must contain a legal GEMV tile"),
                ),
                QmvRoute::MainGemm(plan) => MatmulDispatch::Gemm(plan),
            };
        }
        let gemv = GemvSpecialization::select_shape(
            shape,
            weights_data_type,
            input_data_type,
            output_data_type,
            gpu_core_count,
            apple_gpu_family,
        );
        let problem = GemmProblem::new(*shape, weights_data_type, output_data_type, supports_mxu, apple_gpu_family);
        let plan = problem.select_plan();
        match gemv {
            None => MatmulDispatch::Gemm(plan),
            Some(_)
                if Self::prefer_gemm_over_gemv(*shape, plan, weights_data_type, input_data_type, output_data_type) =>
            {
                MatmulDispatch::Gemm(plan)
            },
            Some(gemv) => MatmulDispatch::Gemv(gemv),
        }
    }

    fn select_dispatch(
        &self,
        shape: &MatmulShape,
        context: &MetalContext,
    ) -> MatmulDispatch {
        Self::choose_dispatch(
            shape,
            &context.device_name,
            context.gpu_core_count,
            context.apple_gpu_family,
            context.supports_mxu,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
        )
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
        let gemv = GemvKernel::new(weights_data_type, input_data_type, output_data_type);

        Ok(Self {
            gemv,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn select_activation_quantization(
        &self,
        shape: &MatmulShape,
        context: &MetalContext,
    ) -> Option<ActivationQuantization> {
        let weight_group_size = shape.b_group_size?;
        let emit_group_sums = match shape.b_prologue {
            GemmBPrologueKind::ScaleSymmetricDequant => false,
            GemmBPrologueKind::ScaleBiasDequant | GemmBPrologueKind::ScaleZeroPointDequant => true,
            GemmBPrologueKind::FullPrecision => return None,
        };
        let code_layout = shape.b_bits.and_then(Int8CodeLayout::for_right_bits)?;
        let quantization =
            ActivationQuantization::new(ACTIVATION_SCALE_GROUP_SIZE, weight_group_size, emit_group_sums, code_layout)?;
        if !context.supports_mxu
            || self.input_data_type != DataType::BF16
            || self.output_data_type != DataType::BF16
            || shape.a_full_precision
            || !shape.is_quant()
            || shape.params_layout != Some(QuantParamsLayout::GroupOutput)
            || !supports_integer_right_operand(shape)
            || !shape.b_transpose
            || shape.b_leading_dimension.is_some()
            || !shape.k.is_multiple_of(ACTIVATION_SCALE_GROUP_SIZE)
            || !shape.k.is_multiple_of(weight_group_size)
        {
            return None;
        }

        Some(quantization)
    }

    fn select_activation_format(
        &self,
        bf16_shape: &MatmulShape,
        context: &MetalContext,
    ) -> ActivationFormat {
        if bf16_shape.params_layout != Some(QuantParamsLayout::GroupOutput)
            || !supports_integer_right_operand(bf16_shape)
            || matches!(self.select_dispatch(bf16_shape, context), MatmulDispatch::Gemv(_))
        {
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

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = match self.select_dispatch(&shape, command_buffer.context()) {
            MatmulDispatch::Gemv(gemv) => {
                return self.gemv.encode(arguments, gemv, command_buffer).map_err(MetalError::from);
            },
            MatmulDispatch::Gemm(plan) => plan,
        };

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
        self.gemm.encode_plan(arguments, plan, command_buffer)
    }
}
