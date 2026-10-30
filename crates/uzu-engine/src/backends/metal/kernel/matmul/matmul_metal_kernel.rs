use metal::MTLGPUFamily;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CpuBuffer,
            kernel::{
                ActivationQuantization,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{
                    ActivationFormat, Int8CodeLayout, MatmulArguments, MatmulKernel, MatmulShape, QuantParamsLayout,
                },
            },
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            gpu_types::gemm::{GemmBLayout, GemmTiling},
            kernel::matmul::{
                GemmKernel, MatmulMetalError,
                gemm::{self, GemmPlan, GemmProblem, small_batch_format, small_m_weights},
                gemv::{GemvKernel, GemvSpecialization, GemvTile},
                matmul_dispatch::MatmulDispatch,
                matmul_output_work::MatmulOutputWork,
            },
        },
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvKernel,
    pub gemm: GemmKernel,
    output_work: MatmulOutputWork,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

pub fn supports_integer_right_operand(shape: &MatmulShape) -> bool {
    matches!((shape.b_bits, shape.signed_codes), (Some(4), _) | (Some(8), true))
}

impl MatmulMetalKernel {
    pub fn prefer_gemm_over_gemv(
        shape: MatmulShape,
        plan: GemmPlan,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> bool {
        // F32 is compiled for GEMV only.
        if shape.gathered || plan.engine != gemm::GemmEngine::Mxu || weights_data_type == DataType::F32 {
            return false;
        }
        if shape.m == 5
            && (weights_data_type, input_data_type, output_data_type)
                == (DataType::BF16, DataType::BF16, DataType::BF16)
        {
            return false;
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
        matches!(
            plan.tiling,
            GemmTiling::Tile16x32x256_Simdgroups1x1
                | GemmTiling::Tile16x32x256_Simdgroups1x1x4
                | GemmTiling::Tile16x128x256_Simdgroups1x4
        )
    }

    pub fn choose_dispatch(
        shape: &MatmulShape,
        gemm_b_layout: GemmBLayout,
        gpu_core_count: u32,
        apple_gpu_family: MTLGPUFamily,
        supports_mxu: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> MatmulDispatch {
        let problem = GemmProblem::new(*shape, weights_data_type, output_data_type, supports_mxu, apple_gpu_family);
        if shape.b_is_trellis {
            return MatmulDispatch::Gemm(problem.select_plan());
        }
        // Small M, one rule for every chip: batched GEMV for M=2..3 (M=2..4 with MXU), the register GEMM above that.
        if all_bf16(weights_data_type, input_data_type, output_data_type) && small_m_weights(shape).is_some() {
            if !supports_mxu && let Some(plan) = problem.small_batch_plan(gemm_b_layout) {
                return MatmulDispatch::Gemm(plan);
            }
            let tile = match (supports_mxu, shape.m) {
                (false, 2..=8) => Some(GemvTile::quantized_output_tile(4, 32, shape.m.min(4), 8, 1)),
                (true, 2..=4) => Some(GemvTile::quantized_output_tile(2, 16, shape.m, 8, 1)),
                _ => None,
            };
            if let Some(gemv) = tile.and_then(|tile| GemvSpecialization::select_tile(shape, gemm_b_layout, tile)) {
                return MatmulDispatch::Gemv(gemv);
            }
        }
        let gemv = GemvSpecialization::select_shape(
            shape,
            gemm_b_layout,
            weights_data_type,
            input_data_type,
            output_data_type,
            gpu_core_count,
        );
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
        gemm_b_layout: GemmBLayout,
        context: &MetalContext,
    ) -> MatmulDispatch {
        Self::choose_dispatch(
            shape,
            gemm_b_layout,
            context.gpu_core_count,
            context.apple_gpu_family,
            context.supports_mxu,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
        )
    }
}

fn all_bf16(
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
) -> bool {
    (weights_data_type, input_data_type, output_data_type) == (DataType::BF16, DataType::BF16, DataType::BF16)
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;
    type Error = MatmulMetalError;
    type WeightsLayout = GemmBLayout;

    fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, Self::Error> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(MatmulMetalError::UnsupportedDataType(data_type));
            }
        }

        let output_work = MatmulOutputWork::new(context, weights_data_type, output_data_type)?;
        let gemm = GemmKernel::new(weights_data_type, input_data_type, output_data_type);
        let gemv = GemvKernel::new(weights_data_type, input_data_type, output_data_type);

        Ok(Self {
            gemv,
            gemm,
            output_work,
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
        let emit_group_sums = GemmKernel::requires_activation_group_sums(shape);
        let code_layout = shape.b_bits.and_then(Int8CodeLayout::for_right_bits)?;
        let quantization =
            ActivationQuantization::new(ACTIVATION_SCALE_GROUP_SIZE, weight_group_size, emit_group_sums, code_layout)?;
        if !context.supports_mxu
            || self.input_data_type != DataType::BF16
            || self.output_data_type != DataType::BF16
            || !shape.is_quant()
            || shape.params_layout != Some(QuantParamsLayout::GroupOutput)
            || !supports_integer_right_operand(shape)
            || !shape.b_transpose
            || shape.b_leading_dimension.is_some()
            || !shape.k.is_multiple_of(ACTIVATION_SCALE_GROUP_SIZE)
            || !shape.k.is_multiple_of(weight_group_size)
            || !shape.n.is_multiple_of(32)
            || shape.b_bits.and_then(|bits| GemmKernel::mxu_tiled_words_per_thread(shape.k, bits)).is_none()
        {
            return None;
        }

        Some(quantization)
    }

    /// Load-time: the activation format this linear prepares once (Int8 on MXU, PackedHalf for the small-batch W4
    /// GEMM, else Bf16). `select_activation_format` decides per batch size whether it is used.
    fn activation_preparation(
        &self,
        shape: &MatmulShape,
        weights_layout: GemmBLayout,
        context: &MetalContext,
    ) -> ActivationFormat {
        if let Some(quantization) = self.select_activation_quantization(shape, context) {
            return ActivationFormat::Int8(quantization);
        }
        match small_batch_format(*shape, weights_layout) {
            Some((format, _))
                if !context.supports_mxu
                    && shape.params_layout == Some(QuantParamsLayout::GroupOutput)
                    && all_bf16(self.weights_data_type, self.input_data_type, self.output_data_type) =>
            {
                format
            },
            _ => ActivationFormat::Bf16,
        }
    }

    /// Encode-time: `prepared` when this batch size has a kernel for it, else Bf16 (GEMV keeps Bf16 input).
    fn select_activation_format(
        &self,
        bf16_shape: &MatmulShape,
        weights_layout: GemmBLayout,
        prepared: ActivationFormat,
        context: &MetalContext,
    ) -> ActivationFormat {
        let dispatch = |shape: &MatmulShape| self.select_dispatch(shape, weights_layout, context);
        let usable = match prepared {
            ActivationFormat::PackedHalf {
                ..
            } => matches!(
                dispatch(&MatmulShape { a: prepared, ..*bf16_shape }),
                MatmulDispatch::Gemm(plan) if plan.tiling.is_small_batch_variant()
            ),
            ActivationFormat::Int8(_) => matches!(dispatch(bf16_shape), MatmulDispatch::Gemm(_)),
            _ => false,
        };
        if usable {
            prepared
        } else {
            ActivationFormat::Bf16
        }
    }

    fn optimize_weights_layout(
        &self,
        a8_shape: &MatmulShape,
        weights: &mut <Metal as Backend>::GlobalGpuBuffer,
        context: &MetalContext,
    ) -> Result<(GemmBLayout, bool), MetalError> {
        let prepared_shape = MatmulShape {
            signed_codes: a8_shape.signed_codes || a8_shape.b_bits == Some(8),
            ..*a8_shape
        };
        if self.select_activation_quantization(&prepared_shape, context).is_none() {
            return Ok((GemmBLayout::RowMajor, a8_shape.signed_codes));
        }
        if !a8_shape.signed_codes && prepared_shape.signed_codes {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    weights.cpu_ptr().as_ptr().cast::<u8>(),
                    crate::backends::common::Buffer::size(weights),
                )
            };
            let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(bytes);
            words.iter_mut().for_each(|word| *word ^= 0x8080_8080_8080_8080);
            prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= 0x80);
        }
        GemmKernel::prepare_gemm_b_layout(
            weights,
            a8_shape.n,
            a8_shape.k,
            a8_shape.b_bits.expect("quantized weights"),
            context,
        )?;
        Ok((GemmBLayout::MxuTiled, prepared_shape.signed_codes))
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
    ) -> Result<(), Self::Error> {
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = match self.select_dispatch(&shape, arguments.b.weights_layout(), command_buffer.context()) {
            MatmulDispatch::Gemv(gemv) => {
                return self.gemv.encode(arguments, gemv, &self.output_work, command_buffer);
            },
            MatmulDispatch::Gemm(plan) => plan,
        };

        self.gemm.encode_plan(arguments, plan, &self.output_work, command_buffer)
    }
}

#[cfg(test)]
impl MatmulMetalKernel {
    pub fn encode_with_gemm_engine(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        engine: gemm::GemmEngine,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MatmulMetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        let plan = self.gemm.select_plan_for_engine(&shape, engine, command_buffer.context())?;
        self.gemm.encode_plan(arguments, plan, &self.output_work, command_buffer)
    }
}
