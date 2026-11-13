use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::{
            gemm::GemmBPrologueKind, hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE, moe, quantization::QuantizationMode,
        },
        kernel::{
            Kernels, MoeCombineKernel, MoeGemvKernel, MoeRouterTopKKernel,
            matmul::{ActivationFormat, MatmulA, MatmulB},
        },
    },
    config::{
        mlp::mixture_of_experts::MixtureOfExpertsConfig,
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            hybrid_spec::{AnyIncoherenceProcessorSpec, HadamardSpec, HybridSpec, IncoherenceProcessingMode},
        },
    },
    data_type::DataType,
    encodable_block::{
        linear::{InputRht, Linear, LinearInputPreparation, LinearMatmul, LinearMatmulError, UntiedReadout},
        mlp::{
            Mlp, MlpEncodeError,
            dense::{DenseMlp, DenseMlpNewError},
            gate_act_mul::MlpGateActMulEncodable,
        },
        weight_matrix::{WeightMatrix, WeightMatrixError},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
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
    input_rht: InputRht<B>,
    routed_up: RoutedProjection<B>,
    routed_activation: MlpGateActMulEncodable<B>,
    routed_down: RoutedProjection<B>,
    shared_expert: DenseMlp<B>,
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
        let num_routed_experts = config.num_routed_experts;
        let num_active_routed_experts = config.num_active_routed_experts;
        let expert_hidden_dim = config.expert_hidden_dim;
        let expert_config = &config.expert_config;

        let router_weights_tree = parameter_tree.subtree("router").subtree("weights");
        let router_spec = router_weights_tree.metadata("spec")?;
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
        let (routed_up, _) = RoutedProjection::load(
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
            &parameter_tree.subtree("shared_experts").stack_of(config.num_shared_experts),
            context,
        )
        .await?;
        let Some(shared_preparation) = shared_preparation else {
            return Err(MoeMlpNewError::UnsupportedConfiguration("shared expert requires input rotation".into()));
        };

        let gate_weights_tree = parameter_tree.subtree("gate").subtree("weights");
        let gate_spec = gate_weights_tree.metadata("spec")?;
        let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            quantization_spec,
            incoherence_spec:
                Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                    block_size: HADAMARD_TRANSFORM_BLOCK_SIZE,
                    processing_mode: IncoherenceProcessingMode::Input,
                    ..
                })),
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

        let router_top_k = <B::Kernels as Kernels>::MoeRouterTopKKernel::new(context, DataType::F32)
            .map_err(MoeMlpNewError::Backend)?;
        let input_rht =
            InputRht::new(data_type, shared_preparation, false, context).map_err(MoeMlpNewError::Backend)?;
        let combine = <B::Kernels as Kernels>::MoeCombineKernel::new(context, data_type, moe::COMBINE_BLOCK_DIM)
            .map_err(MoeMlpNewError::Backend)?;

        Ok(Self {
            router,
            router_top_k,
            input_rht,
            routed_up,
            routed_activation,
            routed_down,
            shared_expert,
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
        let quantized_shared_input = (shared_format != ActivationFormat::Bf16)
            .then(|| self.input_rht.prepare(&input, batch_dim, shared_format, command_buffer))
            .transpose()
            .map_err(MlpEncodeError::Backend)?;
        let MatmulA::FullPrecision {
            values: rotated_input,
            ..
        } = self
            .input_rht
            .prepare(&input, batch_dim, ActivationFormat::Bf16, command_buffer)
            .map_err(MlpEncodeError::Backend)?
        else {
            unreachable!()
        };

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
            self.shared_gate.encode(&rotated_input, batch_dim, 1, None::<&B::ScratchBuffer>, command_buffer)?;
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
    weights: WeightMatrix<B>,
    output_hadamard_factors: B::GlobalGpuBuffer,
    input_dim: u32,
    output_dim: u32,
    data_type: DataType,
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
            incoherence_spec:
                Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                    block_size: HADAMARD_TRANSFORM_BLOCK_SIZE,
                    processing_mode: IncoherenceProcessingMode::InputOutput,
                    ..
                })),
            ..
        }) = spec
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{spec:?}")));
        };
        let weights = WeightMatrix::load(
            output_dim,
            input_dim,
            data_type,
            *quantization_spec,
            Layout::OutputInput,
            &weights_tree.subtree("quantized").stack_of(num_routed_experts),
        )
        .await?;
        let MatmulB::Quantized(quantized) = weights.matmul_b() else {
            return Err(MoeMlpNewError::UnsupportedConfiguration("routed experts require quantized weights".into()));
        };
        match (quantized.mode, quantized.prologue()) {
            (QuantizationMode::U4, GemmBPrologueKind::ScaleZeroPointDequant)
            | (QuantizationMode::U8, GemmBPrologueKind::ScaleSymmetricDequant) => {},
            (mode, prologue) => {
                return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{mode} {prologue}")));
            },
        }
        let kernel = <B::Kernels as Kernels>::MoeGemvKernel::new(
            context,
            data_type,
            quantized.group_size,
            quantized.bits(),
            a_per_token,
        )
        .map_err(MoeMlpNewError::Backend)?;

        let input_hadamard_factors =
            weights_tree.read_buffer("incoherence_signs.input_signs", &[input_dim], DataType::I32).await?;
        let output_hadamard_factors = weights_tree
            .read_buffer("incoherence_signs.output_signs", &[num_routed_experts, output_dim], DataType::I32)
            .await?;

        Ok((
            Self {
                kernel,
                weights,
                output_hadamard_factors,
                input_dim,
                output_dim,
                data_type,
            },
            input_hadamard_factors,
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
        let MatmulB::Quantized(weights) = self.weights.matmul_b() else {
            unreachable!()
        };
        let [scale_rows, scale_columns] = weights.params.scale_shape();
        let [zero_point_rows, zero_point_columns] = weights.params.zero_point_shape(weights.mode);
        let scale_strides = weights.params.scale_strides();
        let zero_point_strides = weights.zero_point_strides();
        let mut output = command_buffer.allocate_scratch_for_shape(&[num_routes, self.output_dim], self.data_type)?;
        self.kernel.encode(
            weights.codes,
            weights.scales,
            weights.zero_points(),
            input,
            expert_ids,
            &mut output,
            &self.output_hadamard_factors,
            self.input_dim,
            self.output_dim,
            num_routes,
            num_active_routed_experts,
            self.output_dim * (self.input_dim / weights.mode.packing_divisor()),
            scale_rows * scale_columns,
            zero_point_rows * zero_point_columns,
            scale_strides.output_stride,
            scale_strides.group_stride,
            zero_point_strides.output_stride,
            zero_point_strides.group_stride,
            command_buffer,
        );

        Ok(output)
    }
}
