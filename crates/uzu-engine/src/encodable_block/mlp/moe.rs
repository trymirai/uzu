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
            ActivationTransform, JangtqDownProjectTileKernel,
            JangtqDownReduceCombineKernel, JangtqFusedGateUpKernel, JangtqFusedGateUpTileKernel,
            JangtqHadamard32Kernel, JangtqHadamard32F32Kernel, JangtqSortIndicesKernel,
            JangtqWeightedDownCombineKernel, Kernels, MoeCombineKernel, MoeGemvKernel,
            MoeRouterTopKKernel,
            matmul::{
                ActivationFormat, MatmulA, QuantParams, QuantParamsLayout, QuantParamsStrides, packed_half_bytes,
            },
        },
    },
    config::{
        mlp::{mixture_of_experts::MixtureOfExpertsConfig, routing_function::AnyRoutingFunction},
        weight_matrix::{
            AnyWeightMatrixSpec,
            hybrid_spec::{AnyIncoherenceProcessorSpec, HadamardSpec, HybridSpec, IncoherenceProcessingMode},
            int_spec::IntSpec,
            jangtq2_spec::Jangtq2Spec,
        },
    },
    data_type::DataType,
    encodable_block::{
        linear::{LinearInputPreparation, LinearMatmul, LinearMatmulError, UntiedReadout},
        mlp::{
            Mlp, MlpEncodeError,
            dense::{DenseMlp, DenseMlpNewError},
            gate_act_mul::MlpGateActMulEncodable,
        },
        weight_matrix::WeightMatrixError,
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
    routed: RoutedPath<B>,
    shared_expert: DenseMlp<B>,
    shared_input_quantizer: Option<ActivationTransform<B>>,
    shared_gate: UntiedReadout<B>,
    shared_gate_raw_input: bool,
    model_dimension: u32,
    num_routed_experts: u32,
    num_active_routed_experts: u32,
    data_type: DataType,
}

/// Routed-expert execution paths. `Gemv` is the rotated-basis W4/W8 path shared with
/// upstream; `Jangtq` runs the MiniCPM JANGTQ banks with the reference's raw-bf16-input,
/// f32-intermediate numerics (input rotation and activation fused into the kernels).
enum RoutedPath<B: Backend> {
    Gemv {
        input_rht: ActivationTransform<B>,
        input_hadamard_factors: B::GlobalGpuBuffer,
        up: RoutedProjection<B>,
        activation: MlpGateActMulEncodable<B>,
        down: RoutedProjection<B>,
        combine: <B::Kernels as Kernels>::MoeCombineKernel,
    },
    Jangtq(Box<JangtqPath<B>>),
}

pub struct JangtqPath<B: Backend> {
    hadamard32: <B::Kernels as Kernels>::JangtqHadamard32Kernel,
    hadamard32_f32: <B::Kernels as Kernels>::JangtqHadamard32F32Kernel,
    sort_indices: <B::Kernels as Kernels>::JangtqSortIndicesKernel,
    gate_up: <B::Kernels as Kernels>::JangtqFusedGateUpKernel,
    gate_up_tile: <B::Kernels as Kernels>::JangtqFusedGateUpTileKernel,
    down_combine: <B::Kernels as Kernels>::JangtqWeightedDownCombineKernel,
    down_project_tile: <B::Kernels as Kernels>::JangtqDownProjectTileKernel,
    down_reduce_combine: <B::Kernels as Kernels>::JangtqDownReduceCombineKernel,
    packed_gate: B::GlobalGpuBuffer,
    scales_gate: B::GlobalGpuBuffer,
    packed_up: B::GlobalGpuBuffer,
    scales_up: B::GlobalGpuBuffer,
    packed_down: B::GlobalGpuBuffer,
    scales_down: B::GlobalGpuBuffer,
    intermediate_dim: u32,
    words_gate: u32,
    words_up: u32,
    words_down: u32,
    alpha_gate: f32,
    beta_gate: f32,
    alpha_up: f32,
    beta_up: f32,
    alpha_down: f32,
    beta_down: f32,
    limit: f32,
}

impl<B: Backend> MoeMlp<B> {
    /// Router matmul + top-k selection. Consumes `input` (see `Linear::encode`).
    fn route(
        &self,
        input: &B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), MlpEncodeError<B>> {
        let router_logits = self.router.encode_with_a(
            MatmulA::full_precision(input),
            batch_dim,
            self.num_routed_experts,
            None::<&B::ScratchBuffer>,
            command_buffer,
        )?;
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
        Ok((expert_ids, expert_weights))
    }

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
        let gate_projection_spec =
            routed_tree.subtree("gate_projection").subtree("weights").metadata::<AnyWeightMatrixSpec>("spec").ok();
        let jangtq_banks: Option<(Jangtq2Spec, Jangtq2Spec, Jangtq2Spec)> = match gate_projection_spec {
            Some(AnyWeightMatrixSpec::Jangtq2Spec(gate_book)) => {
                let up_book = match routed_tree.subtree("up_projection").subtree("weights").metadata("spec")? {
                    AnyWeightMatrixSpec::Jangtq2Spec(book) => book,
                    other => {
                        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                            "JANGTQ gate_projection requires a Jangtq2Spec up_projection, got {other:?}"
                        )));
                    },
                };
                let down_book = match routed_tree.subtree("down_projection").subtree("weights").metadata("spec")? {
                    AnyWeightMatrixSpec::Jangtq2Spec(book) => book,
                    other => {
                        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                            "JANGTQ gate_projection requires a Jangtq2Spec down_projection, got {other:?}"
                        )));
                    },
                };
                Some((gate_book, up_book, down_book))
            },
            _ => None,
        };

        let routed = if let Some((gate_book, up_book, down_book)) = jangtq_banks {
            let load_bank =
                async |name: &str,
                       book: &Jangtq2Spec,
                       n_out: u32,
                       k_in: u32|
                       -> Result<(B::GlobalGpuBuffer, B::GlobalGpuBuffer, u32), MoeMlpNewError<B>> {
                    if !matches!(book.bits, 2 | 3) {
                        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                            "JANGTQ bits must be 2 or 3, got {}",
                            book.bits
                        )));
                    }
                    if !k_in.is_multiple_of(512) || !n_out.is_multiple_of(32) {
                        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                            "JANGTQ bank {name}: k_in {k_in} must be a multiple of 512 and n_out {n_out} a multiple of 32"
                        )));
                    }
                    let words = k_in * book.bits / 32;
                    let quantized = routed_tree.subtree(name).subtree("weights").subtree("quantized");
                    let packed =
                        quantized.read_buffer("weights", &[num_routed_experts, n_out, words], DataType::U32).await?;
                    let scales = quantized.read_buffer("scales", &[num_routed_experts, n_out], DataType::F16).await?;
                    Ok((packed, scales, words))
                };
            let (packed_gate, scales_gate, words_gate) =
                load_bank("gate_projection", &gate_book, expert_hidden_dim, model_dimension).await?;
            let (packed_up, scales_up, words_up) =
                load_bank("up_projection", &up_book, expert_hidden_dim, model_dimension).await?;
            let (packed_down, scales_down, words_down) =
                load_bank("down_projection", &down_book, model_dimension, expert_hidden_dim).await?;
            let limit = match (expert_config.gate_clipping.into_pair(), expert_config.up_clipping.into_pair()) {
                (None, None) => 0.0,
                (gate, up) => {
                    return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                        "JANGTQ activation clipping is not supported (gate {gate:?}, up {up:?})"
                    )));
                },
            };
            let hadamard32 =
                <B::Kernels as Kernels>::JangtqHadamard32Kernel::new(context).map_err(MoeMlpNewError::Backend)?;
            let hadamard32_f32 =
                <B::Kernels as Kernels>::JangtqHadamard32F32Kernel::new(context).map_err(MoeMlpNewError::Backend)?;
            let sort_indices =
                <B::Kernels as Kernels>::JangtqSortIndicesKernel::new(context).map_err(MoeMlpNewError::Backend)?;
            let gate_up = <B::Kernels as Kernels>::JangtqFusedGateUpKernel::new(context, gate_book.bits, up_book.bits)
                .map_err(MoeMlpNewError::Backend)?;
            let gate_up_tile =
                <B::Kernels as Kernels>::JangtqFusedGateUpTileKernel::new(context, gate_book.bits, up_book.bits)
                    .map_err(MoeMlpNewError::Backend)?;
            let down_combine =
                <B::Kernels as Kernels>::JangtqWeightedDownCombineKernel::new(context, data_type, down_book.bits)
                    .map_err(MoeMlpNewError::Backend)?;
            let down_project_tile =
                <B::Kernels as Kernels>::JangtqDownProjectTileKernel::new(context, down_book.bits)
                    .map_err(MoeMlpNewError::Backend)?;
            let down_reduce_combine =
                <B::Kernels as Kernels>::JangtqDownReduceCombineKernel::new(context, data_type, moe::COMBINE_BLOCK_DIM)
                    .map_err(MoeMlpNewError::Backend)?;
            RoutedPath::Jangtq(Box::new(JangtqPath {
                hadamard32,
                hadamard32_f32,
                sort_indices,
                gate_up,
                gate_up_tile,
                down_combine,
                down_project_tile,
                down_reduce_combine,
                packed_gate,
                scales_gate,
                packed_up,
                scales_up,
                packed_down,
                scales_down,
                intermediate_dim: expert_hidden_dim,
                words_gate,
                words_up,
                words_down,
                alpha_gate: gate_book.alpha,
                beta_gate: gate_book.beta,
                alpha_up: up_book.alpha,
                beta_up: up_book.beta,
                alpha_down: down_book.alpha,
                beta_down: down_book.beta,
                limit,
            }))
        } else {
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
            let input_rht =
                ActivationTransform::input_rht(context, data_type, false).map_err(MoeMlpNewError::Backend)?;
            let combine = <B::Kernels as Kernels>::MoeCombineKernel::new(context, data_type, moe::COMBINE_BLOCK_DIM)
                .map_err(MoeMlpNewError::Backend)?;
            RoutedPath::Gemv {
                input_rht,
                input_hadamard_factors,
                up: routed_up,
                activation: routed_activation,
                down: routed_down,
                combine,
            }
        };

        let (shared_expert, shared_input_quantizer) = if matches!(routed, RoutedPath::Jangtq(_)) {
            let (shared_expert, shared_preparation) = DenseMlp::new(
                model_dimension,
                expert_hidden_dim,
                data_type,
                expert_config,
                &parameter_tree.subtree("shared_experts"),
                context,
            )
            .await?;
            if shared_preparation.is_some() {
                return Err(MoeMlpNewError::UnsupportedConfiguration(
                    "JANGTQ models use raw bf16 shared-expert input; quantized input preparation is unexpected".into(),
                ));
            }
            (shared_expert, None)
        } else {
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
            let quantizer = (shared_preparation.activation_format != ActivationFormat::Bf16)
                .then(|| ActivationTransform::quantize(context, data_type, shared_preparation.activation_format))
                .transpose()
                .map_err(MoeMlpNewError::Backend)?;
            (shared_expert, quantizer)
        };

        let gate_weights_tree = parameter_tree.subtree("gate").subtree("weights");
        let gate_spec = gate_weights_tree.metadata("spec")?;
        let (shared_gate, shared_gate_raw_input) = match gate_spec {
            spec @ AnyWeightMatrixSpec::FullPrecisionSpec(_) => {
                let readout =
                    UntiedReadout::load(1, model_dimension, data_type, spec, &gate_weights_tree, context).await?;
                (readout, true)
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                incoherence_spec:
                    Some(AnyIncoherenceProcessorSpec::HadamardSpec(HadamardSpec {
                        block_size: HADAMARD_TRANSFORM_BLOCK_SIZE,
                        processing_mode: IncoherenceProcessingMode::Input,
                        ..
                    })),
                ..
            }) => {
                // NOTE: Assumed all experts and the gate share one input sign row, so rotate input once.
                gate_weights_tree.ignore("incoherence_signs.input_signs", &[model_dimension], DataType::I32)?;
                let readout = UntiedReadout::load(
                    1,
                    model_dimension,
                    data_type,
                    *quantization_spec,
                    &gate_weights_tree.subtree("quantized"),
                    context,
                )
                .await?;
                (readout, false)
            },
            other => {
                return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{other:?}")));
            },
        };

        let router_top_k =
            <B::Kernels as Kernels>::MoeRouterTopKKernel::new(context, DataType::F32, moe::ROUTER_MAX_TOP_K)
                .map_err(MoeMlpNewError::Backend)?;

        Ok(Self {
            router,
            router_top_k,
            routed,
            shared_expert,
            shared_input_quantizer,
            shared_gate,
            shared_gate_raw_input,
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

        let output = match &self.routed {
            RoutedPath::Gemv {
                input_rht,
                input_hadamard_factors,
                up,
                activation,
                down,
                combine,
            } => {
                let shared_format =
                    self.shared_expert.select_activation_format(batch_dim, None, command_buffer.context());
                let quantized_shared_input = if let Some(quantizer) =
                    self.shared_input_quantizer.as_ref().filter(|quantizer| quantizer.format() == shared_format)
                {
                    Some(match shared_format {
                        ActivationFormat::PackedHalf {
                            block_k,
                            bits,
                        } => {
                            let mut values = command_buffer
                                .allocate_scratch(packed_half_bytes(self.model_dimension, block_k, bits))
                                .map_err(MlpEncodeError::Backend)?;
                            quantizer.encode_pack_half(
                                &input,
                                &mut values,
                                input_hadamard_factors,
                                batch_dim,
                                self.model_dimension,
                                command_buffer,
                            );
                            MatmulA::PackedHalf {
                                values,
                                block_k,
                                bits,
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
                                input_hadamard_factors,
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
                input_rht.encode_fp(
                    &input,
                    &mut rotated_input,
                    input_hadamard_factors,
                    batch_dim,
                    self.model_dimension,
                    command_buffer,
                );

                let (expert_ids, expert_weights) = self.route(&input, batch_dim, command_buffer)?;
                let fused_up = up
                    .encode(&rotated_input, &expert_ids, num_routes, self.num_active_routed_experts, command_buffer)
                    .map_err(MlpEncodeError::Backend)?;
                let down_input = activation
                    .encode_for_linear(&fused_up, num_routes, ActivationFormat::Bf16, command_buffer)
                    .map_err(MlpEncodeError::Backend)?;
                let MatmulA::FullPrecision {
                    values: down_input,
                    ..
                } = down_input
                else {
                    unreachable!()
                };
                let routed_outputs = down
                    .encode(&down_input, &expert_ids, num_routes, self.num_active_routed_experts, command_buffer)
                    .map_err(MlpEncodeError::Backend)?;

                debug_assert!(!self.shared_gate_raw_input, "Gemv path consumes its input before the gate");
                let shared_gate_logits = self.shared_gate.encode(
                    &rotated_input,
                    batch_dim,
                    1,
                    None::<&B::ScratchBuffer>,
                    command_buffer,
                )?;
                let shared_input = quantized_shared_input.unwrap_or_else(|| MatmulA::full_precision(rotated_input));
                let shared_output = self.shared_expert.encode_input(shared_input, batch_dim, command_buffer)?;

                let mut output = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], self.data_type)
                    .map_err(MlpEncodeError::Backend)?;
                combine.encode(
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
                output
            },
            RoutedPath::Jangtq(path) => {
                // Reference numerics: raw bf16 input rotated to f32 per 32-vector, fused
                // gate/up in f32 with the down-basis output rotation, route-weighted f32
                // down projection cast once to bf16, MLX-ordered bf16 combine.
                let mut rotated_input = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], DataType::F32)
                    .map_err(MlpEncodeError::Backend)?;
                path.hadamard32.encode(&input, &mut rotated_input, self.model_dimension, batch_dim, command_buffer);

                // The shared-expert gate runs on the raw bf16 input before routing consumes it.
                let shared_gate_logits = self.shared_gate.encode(&input, batch_dim, 1, None::<&B::ScratchBuffer>, command_buffer)?;

                let (expert_ids, expert_weights) = self.route(&input, batch_dim, command_buffer)?;

                let dispatches = num_routes;
                let (sorted_indices, expert_offsets, expert_counts) = if batch_dim > 1 {
                    let mut sorted = command_buffer
                        .allocate_scratch_for_shape(&[dispatches], DataType::U32)
                        .map_err(MlpEncodeError::Backend)?;
                    let mut offsets = command_buffer
                        .allocate_scratch_for_shape(&[256], DataType::U32)
                        .map_err(MlpEncodeError::Backend)?;
                    let mut counts = command_buffer
                        .allocate_scratch_for_shape(&[256], DataType::U32)
                        .map_err(MlpEncodeError::Backend)?;
                    path.sort_indices.encode(
                        &expert_ids,
                        &mut sorted,
                        &mut offsets,
                        &mut counts,
                        dispatches,
                        command_buffer,
                    );
                    (Some(sorted), Some(offsets), Some(counts))
                } else {
                    (None, None, None)
                };
                let gate_up_indices = sorted_indices.as_ref().unwrap_or(&expert_ids);

                let mut fused_hidden = command_buffer
                    .allocate_scratch_for_shape(&[dispatches, path.intermediate_dim], DataType::F32)
                    .map_err(MlpEncodeError::Backend)?;
                if batch_dim > 1 {
                    path.gate_up_tile.encode(
                        &rotated_input,
                        &path.packed_gate,
                        &path.scales_gate,
                        &path.packed_up,
                        &path.scales_up,
                        gate_up_indices,
                        expert_offsets.as_ref().unwrap(),
                        expert_counts.as_ref().unwrap(),
                        path.alpha_gate,
                        path.beta_gate,
                        path.alpha_up,
                        path.beta_up,
                        self.model_dimension,
                        path.intermediate_dim,
                        path.words_gate,
                        path.words_up,
                        self.num_active_routed_experts,
                        dispatches,
                        path.limit,
                        &mut fused_hidden,
                        command_buffer,
                    );
                    path.hadamard32_f32.encode(
                        &mut fused_hidden,
                        path.intermediate_dim,
                        dispatches,
                        command_buffer,
                    );
                } else {
                    path.gate_up.encode(
                        &rotated_input,
                        &path.packed_gate,
                        &path.scales_gate,
                        &path.packed_up,
                        &path.scales_up,
                        gate_up_indices,
                        path.alpha_gate,
                        path.beta_gate,
                        path.alpha_up,
                        path.beta_up,
                        self.model_dimension,
                        path.intermediate_dim,
                        path.words_gate,
                        path.words_up,
                        self.num_active_routed_experts,
                        dispatches,
                        path.limit,
                        &mut fused_hidden,
                        command_buffer,
                    );
                }

                
                if batch_dim > 1 {
                    let mut routed_dispatches = command_buffer
                        .allocate_scratch_for_shape(&[dispatches, self.model_dimension], DataType::F32)
                        .map_err(MlpEncodeError::Backend)?;
                    path.down_project_tile.encode(
                        &fused_hidden,
                        &path.packed_down,
                        &path.scales_down,
                        gate_up_indices,
                        expert_offsets.as_ref().unwrap(),
                        expert_counts.as_ref().unwrap(),
                        &expert_weights,
                        &mut routed_dispatches,
                        path.alpha_down,
                        path.beta_down,
                        path.intermediate_dim,
                        self.model_dimension,
                        path.words_down,
                        self.num_active_routed_experts,
                        dispatches,
                        command_buffer,
                    );

                    let shared_output = self.shared_expert.encode(input, batch_dim, command_buffer)?;

                    let mut output = command_buffer
                        .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], self.data_type)
                        .map_err(MlpEncodeError::Backend)?;
                    path.down_reduce_combine.encode(
                        &routed_dispatches,
                        &shared_output,
                        &shared_gate_logits,
                        &mut output,
                        self.model_dimension,
                        batch_dim,
                        self.num_active_routed_experts,
                        command_buffer,
                    );
                    output
                } else {
                    let shared_output = self.shared_expert.encode(input, batch_dim, command_buffer)?;

                    let mut output = command_buffer
                        .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], self.data_type)
                        .map_err(MlpEncodeError::Backend)?;
                    path.down_combine.encode(
                        &fused_hidden,
                        &path.packed_down,
                        &path.scales_down,
                        &expert_ids,
                        &expert_weights,
                        &shared_output,
                        &shared_gate_logits,
                        &mut output,
                        path.alpha_down,
                        path.beta_down,
                        path.intermediate_dim,
                        self.model_dimension,
                        path.words_down,
                        self.num_active_routed_experts,
                        batch_dim,
                        command_buffer,
                    );
                    output
                }
            },
        };

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
            incoherence_spec: Some(AnyIncoherenceProcessorSpec::HadamardSpec(hadamard)),
            ..
        }) = spec
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{spec:?}")));
        };
        if hadamard.block_size != HADAMARD_TRANSFORM_BLOCK_SIZE
            || hadamard.processing_mode != IncoherenceProcessingMode::Input
        {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{hadamard:?}")));
        }
        let AnyWeightMatrixSpec::IntSpec(IntSpec {
            bits,
            is_symmetric,
            ..
        }) = *quantization_spec
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("{quantization_spec:?}")));
        };
        let Some(mode) = QuantizationMode::from_unsigned_bits(bits) else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!("bits={bits}")));
        };
        let method =
            if is_symmetric { QuantizationMethod::ScaleSymmetric } else { QuantizationMethod::ScaleZeroPoint };
        let input_block = match (mode, method) {
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
        let packing_divisor = mode.packing_divisor();
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
        let storage_type = mode.storage_type();
        let params = QuantParams::new(QuantParamsLayout::GroupOutput, output_dim, input_dim / moe::GEMV_GROUP_SIZE);
        let weights = quantized_tree
            .read_buffer("weights", &[num_routed_experts, output_dim, input_dim / packing_divisor], storage_type)
            .await?;
        let [scale_rows, scale_columns] = params.scale_shape();
        let scales =
            quantized_tree.read_buffer("scales", &[num_routed_experts, scale_rows, scale_columns], data_type).await?;
        let (zero_points, zero_point_expert_stride, zero_point_strides) = if method
            == QuantizationMethod::ScaleZeroPoint
        {
            let [zero_point_rows, zero_point_columns] = params.zero_point_shape(mode);
            let zero_points = quantized_tree
                .read_buffer("zero_points", &[num_routed_experts, zero_point_rows, zero_point_columns], storage_type)
                .await?;
            (Some(zero_points), zero_point_rows * zero_point_columns, params.zero_point_strides(mode))
        } else {
            (None, 0, QuantParamsStrides::default())
        };

        let bits = DataType::from(mode).size_in_bits() as u32;
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
            input_hadamard_factors.handoff().map_err(MoeMlpNewError::Backend)?,
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
