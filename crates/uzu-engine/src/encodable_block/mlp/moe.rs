use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu,
        gpu_types::{
            hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE, moe, quantization::QuantizationMode,
            quantization_method::QuantizationMethod,
        },
        kernel::{
            ActivationTransform, Kernels, MoeCombineKernel, MoeGemvKernel, MoeRouterTopKKernel,
            matmul::{
                ActivationFormat, MatmulA, QuantParams, QuantParamsLayout, QuantParamsStrides, packed_half_bytes,
            },
        },
    },
    config::{
        mlp::{mixture_of_experts::MixtureOfExpertsConfig, routing_function::AnyRoutingFunction},
        weight_matrix::{
            AnyWeightMatrixSpec,
            hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        },
    },
    data_type::DataType,
    encodable_block::{
        linear::{Gather, Linear, LinearInputPreparation, LinearMatmul, LinearMatmulError, UntiedReadout},
        mlp::{
            Mlp, MlpEncodeError,
            dense::{DenseMlp, DenseMlpNewError},
            gate_act_mul::MlpGateActMulEncodable,
        },
        weight_matrix::{ParsedWeightSpec, WeightMatrixError, parse_spec},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::size_for_shape::size_for_shape,
};

#[derive(Debug, Error)]
pub enum MoeMlpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear matmul error: {0}")]
    LinearMatmul(#[from] LinearMatmulError<B>),
    #[error("Dense MLP error: {0}")]
    Dense(#[from] DenseMlpNewError<B>),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct MoeMlp<B: Backend> {
    router: LinearMatmul<B>,
    router_top_k: <B::Kernels as Kernels>::MoeRouterTopKKernel,
    input_rht: ActivationTransform<B>,
    input_hadamard_factors: B::GlobalGpuBuffer,
    routed_up: RoutedProjection<B>,
    routed_activation: MlpGateActMulEncodable<B>,
    routed_down: RoutedProjection<B>,
    shared_expert: DenseMlp<B>,
    shared_input_quantizer: Option<ActivationTransform<B>>,
    shared_gate: UntiedReadout<B>,
    combine: <B::Kernels as Kernels>::MoeCombineKernel,
    model_dimension: u32,
    num_routed_experts: u32,
    num_active_routed_experts: u32,
    data_type: DataType,
}

impl<B: Backend> MoeMlp<B> {
    pub async fn new(
        model_dimension: u32,
        data_type: DataType,
        config: &MixtureOfExpertsConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, MoeMlpNewError<B>> {
        let AnyRoutingFunction::SoftmaxRouting(_) = config.routing_function;
        let num_routed_experts = config.num_routed_experts;
        let num_active_routed_experts = config.num_active_routed_experts;
        let expert_hidden_dim = config.expert_hidden_dim;
        let expert_config = &config.expert_config;
        if num_routed_experts > moe::ROUTER_MAX_EXPERTS {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "num_routed_experts must be at most {max}, got {num_routed_experts}",
                max = moe::ROUTER_MAX_EXPERTS
            )));
        }
        let max_top_k = moe::ROUTER_MAX_TOP_K.min(num_routed_experts);
        if num_active_routed_experts == 0 || num_active_routed_experts > max_top_k {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "num_active_routed_experts must be in 1..={max_top_k}, got {num_active_routed_experts}"
            )));
        }

        let router_weights_tree = parameter_tree.subtree("router").subtree("weights");
        let router_spec = router_weights_tree.metadata("spec")?;
        let AnyWeightMatrixSpec::FullPrecisionSpec(_) = router_spec else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{router_spec:?}")));
        };
        let router = LinearMatmul::load(
            model_dimension,
            num_routed_experts,
            data_type,
            data_type,
            DataType::F32,
            router_spec,
            None,
            &router_weights_tree,
            None,
            context,
        )
        .await?;

        let routed_tree = parameter_tree.subtree("routed_experts");
        let (routed_up, input_hadamard_factors) = RoutedProjection::load(
            num_routed_experts,
            model_dimension,
            2 * expert_hidden_dim,
            data_type,
            true,
            &routed_tree.subtree("up_projection"),
            context,
        )
        .await?;
        let (routed_down, hidden_hadamard_factors) = RoutedProjection::load(
            num_routed_experts,
            expert_hidden_dim,
            model_dimension,
            data_type,
            false,
            &routed_tree.subtree("down_projection"),
            context,
        )
        .await?;
        let routed_activation = MlpGateActMulEncodable::new(
            expert_hidden_dim,
            data_type,
            expert_config.activation.clone(),
            expert_config.gate_clipping,
            expert_config.up_clipping,
            Some(LinearInputPreparation {
                rht_signs: hidden_hadamard_factors,
                activation_format: ActivationFormat::Bf16,
            }),
            context,
        )
        .map_err(MoeMlpNewError::Backend)?;

        let (shared_expert, shared_preparation) = DenseMlp::new_for_fused_input(
            model_dimension,
            expert_hidden_dim,
            data_type,
            expert_config,
            &parameter_tree.subtree("shared_experts"),
            context,
        )
        .await?;
        let Some(shared_preparation) = shared_preparation else {
            return Err(MoeMlpNewError::UnsupportedConfiguration("shared expert requires input rotation".into()));
        };
        let shared_input_quantizer = (shared_preparation.activation_format != ActivationFormat::Bf16)
            .then(|| ActivationTransform::quantize(context, data_type, shared_preparation.activation_format))
            .transpose()
            .map_err(MoeMlpNewError::Backend)?;

        let gate_weights_tree = parameter_tree.subtree("gate").subtree("weights");
        let gate_spec = gate_weights_tree.metadata("spec")?;
        let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            quantization_spec,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            incoherence_processing_mode: IncoherenceProcessingMode::Input,
            ..
        }) = gate_spec
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{gate_spec:?}")));
        };
        // NOTE: Assumed all experts and the gate share one input sign row, so rotate input once.
        gate_weights_tree.ignore("incoherence_signs.input_signs", &[model_dimension], DataType::I32)?;
        let shared_gate = UntiedReadout::load(
            1,
            model_dimension,
            data_type,
            *quantization_spec,
            &gate_weights_tree.subtree("quantized"),
            context,
        )
        .await?;

        let router_top_k =
            <B::Kernels as Kernels>::MoeRouterTopKKernel::new(context, DataType::F32, moe::ROUTER_MAX_TOP_K)
                .map_err(MoeMlpNewError::Backend)?;
        let input_rht = ActivationTransform::input_rht(context, data_type, false).map_err(MoeMlpNewError::Backend)?;
        let combine = <B::Kernels as Kernels>::MoeCombineKernel::new(context, data_type, moe::COMBINE_BLOCK_DIM)
            .map_err(MoeMlpNewError::Backend)?;

        Ok(Self {
            router,
            router_top_k,
            input_rht,
            input_hadamard_factors,
            routed_up,
            routed_activation,
            routed_down,
            shared_expert,
            shared_input_quantizer,
            shared_gate,
            combine,
            model_dimension,
            num_routed_experts,
            num_active_routed_experts,
            data_type,
        })
    }
}

impl<B: Backend> Mlp<B> for MoeMlp<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (moe)");
        let num_routes = batch_dim * self.num_active_routed_experts;

        let shared_format = self.shared_expert.select_activation_format(batch_dim, command_buffer.context());
        let quantized_shared_input = if let Some(quantizer) =
            self.shared_input_quantizer.as_ref().filter(|quantizer| quantizer.format() == shared_format)
        {
            Some(match shared_format {
                ActivationFormat::PackedHalf {
                    group_size,
                } => {
                    let mut values = command_buffer
                        .allocate_scratch(packed_half_bytes(self.model_dimension, group_size))
                        .map_err(MlpEncodeError::Backend)?;
                    quantizer.encode_pack_half(
                        &input,
                        &mut values,
                        &self.input_hadamard_factors,
                        batch_dim,
                        self.model_dimension,
                        command_buffer,
                    );
                    MatmulA::PackedHalf {
                        values,
                        group_size,
                    }
                },
                ActivationFormat::Int8(quantization) => {
                    let mut values = command_buffer
                        .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], DataType::I8)
                        .map_err(MlpEncodeError::Backend)?;
                    let mut scales = command_buffer
                        .allocate_scratch_for_shape(
                            &[batch_dim, self.model_dimension.div_ceil(quantization.scale_group_size())],
                            DataType::F32,
                        )
                        .map_err(MlpEncodeError::Backend)?;
                    let mut group_sums = quantization
                        .sum_group_size()
                        .map(|group_size| {
                            command_buffer.allocate_scratch_for_shape(
                                &[batch_dim, self.model_dimension.div_ceil(group_size)],
                                DataType::I32,
                            )
                        })
                        .transpose()
                        .map_err(MlpEncodeError::Backend)?;
                    quantizer.encode_quantize(
                        &input,
                        &mut values,
                        &mut scales,
                        group_sums.as_mut(),
                        &self.input_hadamard_factors,
                        batch_dim,
                        self.model_dimension,
                        command_buffer,
                    );
                    MatmulA::Int8Symmetric {
                        values,
                        scales,
                        group_sums,
                        quantization,
                    }
                },
                ActivationFormat::Bf16 | ActivationFormat::Trellis => unreachable!(),
            })
        } else {
            None
        };

        let mut rotated_input = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], self.data_type)
            .map_err(MlpEncodeError::Backend)?;
        self.input_rht.encode_fp(
            &input,
            &mut rotated_input,
            &self.input_hadamard_factors,
            batch_dim,
            self.model_dimension,
            command_buffer,
        );

        let router_logits = self.router.encode(input, batch_dim, command_buffer)?;
        let mut expert_ids = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.num_active_routed_experts], DataType::U32)
            .map_err(MlpEncodeError::Backend)?;
        let mut expert_weights = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.num_active_routed_experts], DataType::F32)
            .map_err(MlpEncodeError::Backend)?;
        self.router_top_k.encode(
            &router_logits,
            &mut expert_ids,
            &mut expert_weights,
            self.num_routed_experts,
            self.num_active_routed_experts,
            batch_dim,
            command_buffer,
        );
        let fused_up = self
            .routed_up
            .encode(&rotated_input, &expert_ids, num_routes, self.num_active_routed_experts, command_buffer)
            .map_err(MlpEncodeError::Backend)?;
        let down_input = self
            .routed_activation
            .encode_for_linear(&fused_up, num_routes, ActivationFormat::Bf16, command_buffer)
            .map_err(MlpEncodeError::Backend)?;
        let MatmulA::FullPrecision {
            values: down_input,
            ..
        } = down_input
        else {
            unreachable!()
        };
        let routed_outputs = self
            .routed_down
            .encode(&down_input, &expert_ids, num_routes, self.num_active_routed_experts, command_buffer)
            .map_err(MlpEncodeError::Backend)?;

        let shared_gate_logits =
            self.shared_gate.encode(&rotated_input, batch_dim, None::<Gather<&B::ScratchBuffer>>, command_buffer)?;
        let shared_input = quantized_shared_input.unwrap_or_else(|| MatmulA::full_precision(rotated_input));
        let shared_output = self.shared_expert.encode_input(shared_input, batch_dim, command_buffer)?;

        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], self.data_type)
            .map_err(MlpEncodeError::Backend)?;
        self.combine.encode(
            &routed_outputs,
            &expert_weights,
            &shared_output,
            &shared_gate_logits,
            &mut output,
            self.model_dimension,
            self.num_active_routed_experts,
            batch_dim,
            command_buffer,
        );

        Ok(output)
    }
}

struct RoutedProjection<B: Backend> {
    kernel: <B::Kernels as Kernels>::MoeGemvKernel,
    weights: B::GlobalGpuBuffer,
    scales: B::GlobalGpuBuffer,
    zero_points: Option<B::GlobalGpuBuffer>,
    output_hadamard_factors: B::GlobalGpuBuffer,
    input_dim: u32,
    output_dim: u32,
    data_type: DataType,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_strides: QuantParamsStrides,
    zero_point_strides: QuantParamsStrides,
}

impl<B: Backend> RoutedProjection<B> {
    async fn load(
        num_routed_experts: u32,
        input_dim: u32,
        output_dim: u32,
        data_type: DataType,
        a_per_token: bool,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, B::GlobalGpuBuffer), MoeMlpNewError<B>> {
        let weights_tree = parameter_tree.subtree("weights");
        let spec = weights_tree.metadata("spec")?;
        let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            quantization_spec,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            ..
        }) = spec
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{spec:?}")));
        };
        let ParsedWeightSpec {
            quantization: Some(quantization),
            ..
        } = parse_spec::<B>(&quantization_spec)?
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{quantization_spec:?}")));
        };
        let input_block = match (quantization.mode, quantization.method) {
            (QuantizationMode::U4, QuantizationMethod::ScaleZeroPoint) => moe::GEMV_W4_INPUT_BLOCK,
            (QuantizationMode::U8, QuantizationMethod::ScaleSymmetric) => moe::GEMV_W8_INPUT_BLOCK,
            (mode, method) => {
                return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{method} {mode}")));
            },
        };
        if !input_dim.is_multiple_of(input_block) {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "input_dim must be a multiple of {input_block}, got {input_dim}"
            )));
        }
        let packing_divisor = quantization.mode.packing_divisor();
        let Some(b_expert_stride_bytes) = output_dim
            .checked_mul(input_dim / packing_divisor)
            .filter(|plane_bytes| (num_routed_experts - 1).checked_mul(*plane_bytes).is_some())
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "{num_routed_experts}x{output_dim}x{input_dim} weights exceed 32-bit offsets"
            )));
        };

        let input_signs =
            weights_tree.read_slice::<i32>("incoherence_signs.input_signs", &[num_routed_experts, input_dim]).await?;
        let mut input_hadamard_factors = context
            .create_buffer_cpu_gpu(size_for_shape(&[input_dim], DataType::I32))
            .map_err(MoeMlpNewError::Backend)?;
        input_hadamard_factors.copyin(&input_signs[..input_dim as usize]);
        let output_hadamard_factors = weights_tree
            .read_buffer("incoherence_signs.output_signs", &[num_routed_experts, output_dim], DataType::I32)
            .await?;

        let quantized_tree = weights_tree.subtree("quantized");
        let storage_type = quantization.mode.storage_type();
        let params = QuantParams::new(QuantParamsLayout::GroupOutput, output_dim, input_dim / moe::GEMV_GROUP_SIZE);
        let weights = quantized_tree
            .read_buffer("weights", &[num_routed_experts, output_dim, input_dim / packing_divisor], storage_type)
            .await?;
        let [scale_rows, scale_columns] = params.scale_shape();
        let scales =
            quantized_tree.read_buffer("scales", &[num_routed_experts, scale_rows, scale_columns], data_type).await?;
        let (zero_points, zero_point_expert_stride, zero_point_strides) = if quantization.method
            == QuantizationMethod::ScaleZeroPoint
        {
            let [zero_point_rows, zero_point_columns] = params.zero_point_shape(quantization.mode);
            let zero_points = quantized_tree
                .read_buffer("zero_points", &[num_routed_experts, zero_point_rows, zero_point_columns], storage_type)
                .await?;
            (Some(zero_points), zero_point_rows * zero_point_columns, params.zero_point_strides(quantization.mode))
        } else {
            (None, 0, QuantParamsStrides::default())
        };

        let bits = DataType::from(quantization.mode).size_in_bits() as u32;
        let kernel =
            <B::Kernels as Kernels>::MoeGemvKernel::new(context, data_type, moe::GEMV_GROUP_SIZE, bits, a_per_token)
                .map_err(MoeMlpNewError::Backend)?;

        Ok((
            Self {
                kernel,
                weights,
                scales,
                zero_points,
                output_hadamard_factors,
                input_dim,
                output_dim,
                data_type,
                b_expert_stride_bytes,
                scale_expert_stride: scale_rows * scale_columns,
                zero_point_expert_stride,
                scale_strides: params.scale_strides(),
                zero_point_strides,
            },
            input_hadamard_factors.handoff(),
        ))
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        expert_ids: impl BufferRef<Backend = B>,
        num_routes: u32,
        num_active_routed_experts: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = command_buffer.allocate_scratch_for_shape(&[num_routes, self.output_dim], self.data_type)?;
        self.kernel.encode(
            &self.weights,
            &self.scales,
            self.zero_points.as_ref(),
            input,
            expert_ids,
            &mut output,
            &self.output_hadamard_factors,
            self.input_dim,
            self.output_dim,
            num_routes,
            num_active_routed_experts,
            self.output_dim / moe::GEMV_OUTPUT_ROW_TILE,
            self.b_expert_stride_bytes,
            self.scale_expert_stride,
            self.zero_point_expert_stride,
            self.scale_strides.output_stride,
            self.scale_strides.group_stride,
            self.zero_point_strides.output_stride,
            self.zero_point_strides.group_stride,
            command_buffer,
        );

        Ok(output)
    }
}
