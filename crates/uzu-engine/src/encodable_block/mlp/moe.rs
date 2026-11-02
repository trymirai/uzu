//! Mixture-of-experts MLP block.

use std::mem::size_of_val;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu,
        gpu_types::{
            activation_type::ActivationType, hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
            quantization::QuantizationMode, quantization_method::QuantizationMethod,
        },
        kernel::{
            ActivationTransform, Kernels, MoeCombineKernel, MoeGemvKernel, MoeRouterTopKKernel,
            matmul::{ActivationFormat, MatmulA, QuantParams, QuantParamsLayout, QuantParamsStrides},
        },
    },
    config::{
        mlp::{mixture_of_experts::MixtureOfExpertsConfig, routing_function::AnyRoutingFunction},
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            full_precision_spec::FullPrecisionSpec,
            hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        },
    },
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError, LinearInputPreparation, LinearMatmul},
        mlp::{
            Mlp, MlpEncodeError,
            dense::{DenseMlp, DenseMlpNewError},
            gate_act_mul::MlpGateActMulEncodable,
        },
        weight_matrix::{ParsedWeightSpec, QuantizationInfo, WeightMatrixError, parse_spec},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

// Limits of the MoE kernels, from backends/metal/kernel/moe/common.h, their VARIANTS lists and the contract above
// MoeGemv: the router scores up to METAL_SIMD_SIZE x MOE_ROUTER_MAX_PER_LANE experts and picks up to MAX_TOP_K, the
// GEMV runs GROUP_SIZE 64 over whole input blocks and output tiles, and the combine runs BD 256.
const MOE_ROUTER_MAX_TOP_K: u32 = 8;
const MOE_ROUTER_MAX_EXPERTS: u32 = 256;
const MOE_GEMV_GROUP_SIZE: u32 = 64;
const MOE_GEMV_W4_INPUT_BLOCK: u32 = 512;
const MOE_GEMV_W8_INPUT_BLOCK: u32 = 256;
const MOE_GEMV_OUTPUT_ROW_TILE: u32 = 32;
const MOE_COMBINE_BLOCK_DIM: u32 = 256;

#[derive(Debug, Error)]
pub enum MoeMlpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Dense MLP error: {0}")]
    Dense(#[from] DenseMlpNewError<B>),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error("Parameter loader error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Unsupported MoE configuration: {0}")]
    UnsupportedConfiguration(String),
}

/// Rejects layers the MoE kernels do not run, before any weight is read. Qwen3.5-35B-A3B routes each token to 8 of 256
/// experts and adds one shared expert scaled by a sigmoid gate, and none of its projections has a bias.
fn check_config<B: Backend>(
    config: &MixtureOfExpertsConfig,
    data_type: DataType,
) -> Result<(), MoeMlpNewError<B>> {
    // New routing functions must be rejected here: the router kernel takes a softmax over the picked scores.
    match config.routing_function {
        AnyRoutingFunction::SoftmaxRouting(_) => {},
    }
    let num_experts = config.num_routed_experts;
    let top_k = config.num_active_routed_experts;
    let expert_config = &config.expert_config;
    let problem = if num_experts > MOE_ROUTER_MAX_EXPERTS {
        format!("num_routed_experts ({num_experts}) is above the router's {MOE_ROUTER_MAX_EXPERTS}")
    } else if top_k == 0 || top_k > MOE_ROUTER_MAX_TOP_K.min(num_experts) {
        format!(
            "num_active_routed_experts ({top_k}) must be at least 1 and at most {MOE_ROUTER_MAX_TOP_K} and num_routed_experts ({num_experts})"
        )
    } else if config.num_shared_experts != 1 || config.gate_config.is_none() {
        format!(
            "expected one shared expert with a gate, got num_shared_experts ({}) and gate_config {:?}",
            config.num_shared_experts, config.gate_config
        )
    } else if config.router_has_biases || expert_config.has_up_biases || expert_config.has_down_biases {
        "router and expert biases are not supported".to_string()
    } else if expert_config.activation.act_type() == ActivationType::IDENTITY {
        "the experts' gated activation cannot be the identity".to_string()
    } else if data_type != DataType::BF16 {
        format!("activations must be BF16, got {data_type:?}")
    } else {
        return Ok(());
    };
    Err(MoeMlpNewError::UnsupportedConfiguration(problem))
}

/// Splits the spec of quantized MoE weights into their quantization spec and format: they must be rotated as `mode`
/// says in blocks of HADAMARD_TRANSFORM_BLOCK_SIZE and stored output_input in groups of MOE_GEMV_GROUP_SIZE, as 4-bit
/// codes with zero points or 8-bit symmetric codes.
fn check_quantized_spec<B: Backend>(
    weights_name: &str,
    spec: AnyWeightMatrixSpec,
    mode: IncoherenceProcessingMode,
) -> Result<(AnyWeightMatrixSpec, QuantizationInfo), MoeMlpNewError<B>> {
    let quantization_spec = match spec {
        AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            quantization_spec,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            incoherence_processing_mode,
            ..
        }) if incoherence_processing_mode == mode => *quantization_spec,
        spec => {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "{weights_name} must rotate {mode:?} in blocks of {HADAMARD_TRANSFORM_BLOCK_SIZE}, got {spec:?}"
            )));
        },
    };
    let ParsedWeightSpec {
        layout,
        quantization,
    } = parse_spec::<B>(&quantization_spec)?;
    let Some(quantization) = quantization
        .filter(|quantization| layout == Layout::OutputInput && quantization.group_size == MOE_GEMV_GROUP_SIZE)
    else {
        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
            "{weights_name} must be quantized output_input in groups of {MOE_GEMV_GROUP_SIZE}, got {quantization_spec:?}"
        )));
    };
    match (quantization.mode, quantization.method) {
        (QuantizationMode::U4, QuantizationMethod::ScaleZeroPoint)
        | (QuantizationMode::U8, QuantizationMethod::ScaleSymmetric) => Ok((quantization_spec, quantization)),
        (mode, method) => Err(MoeMlpNewError::UnsupportedConfiguration(format!(
            "{weights_name} must be 4-bit with zero points or 8-bit symmetric, got {mode} {method}"
        ))),
    }
}

/// How a routed projection's experts are stored, as the MoE GEMV reads them.
struct RoutedFormat {
    quantization: QuantizationInfo,
    bits: u32,
    b_expert_stride_bytes: u32,
}

/// Checks that routed experts stored with `spec` fit the MoE GEMV, which has no bounds checks: whole input blocks and
/// output tiles, and expert planes offset with 32-bit products. `num_experts` is at least 1, as check_config ensures.
fn check_routed_projection<B: Backend>(
    spec: AnyWeightMatrixSpec,
    num_experts: u32,
    input_dim: u32,
    output_dim: u32,
) -> Result<RoutedFormat, MoeMlpNewError<B>> {
    let (_, quantization) = check_quantized_spec("routed experts", spec, IncoherenceProcessingMode::InputOutput)?;
    let bits = DataType::from(quantization.mode).size_in_bits() as u32;
    let input_block = if quantization.mode == QuantizationMode::U4 {
        MOE_GEMV_W4_INPUT_BLOCK
    } else {
        MOE_GEMV_W8_INPUT_BLOCK
    };
    if !input_dim.is_multiple_of(input_block) || !output_dim.is_multiple_of(MOE_GEMV_OUTPUT_ROW_TILE) {
        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
            "{bits}-bit routed experts need input_dim ({input_dim}) a multiple of {input_block} and output_dim ({output_dim}) a multiple of {MOE_GEMV_OUTPUT_ROW_TILE}"
        )));
    }
    // The GEMV offsets an expert's weights, scales, zero points and output signs by 32-bit products of its id, at
    // most num_experts - 1, and the weight plane is the largest of them.
    let Some(b_expert_stride_bytes) = output_dim
        .checked_mul(input_dim / quantization.mode.packing_divisor())
        .filter(|plane_bytes| (num_experts - 1).checked_mul(*plane_bytes).is_some())
    else {
        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
            "{num_experts} expert planes of {output_dim} x {input_dim} {bits}-bit weights exceed 32-bit offsets"
        )));
    };
    Ok(RoutedFormat {
        quantization,
        bits,
        b_expert_stride_bytes,
    })
}

/// Whether every expert's row of `signs` equals the first expert's.
fn experts_share_input_signs(
    signs: &[i32],
    input_dim: usize,
) -> bool {
    let (first_expert_signs, other_experts_signs) = signs.split_at(input_dim);
    other_experts_signs.chunks_exact(input_dim).all(|expert_signs| expert_signs == first_expert_signs)
}

/// Uploads the sign row all experts share. The rotation kernels index signs by column, so one row serves every route.
fn sign_buffer<B: Backend>(
    signs: &[i32],
    context: &B::Context,
) -> Result<B::GlobalGpuBuffer, MoeMlpNewError<B>> {
    let mut buffer = context.create_buffer_cpu_gpu(size_of_val(signs)).map_err(MoeMlpNewError::Backend)?;
    buffer.copyin(signs);
    Ok(buffer.handoff())
}

/// Errors unless the linear under `tree` rotates its input with `routed_input_signs`, so it can read the input the
/// routed experts rotate.
async fn check_reads_routed_rotation<B: Backend>(
    tree: &ParameterTree<'_, B>,
    routed_input_signs: &[i32],
    linear_name: &str,
) -> Result<(), MoeMlpNewError<B>> {
    let input_signs = tree
        .subtree("weights")
        .read_slice::<i32>("incoherence_signs.input_signs", &[routed_input_signs.len() as u32])
        .await?;
    if *input_signs != *routed_input_signs {
        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
            "the {linear_name} must rotate its input with the routed experts' input signs"
        )));
    }
    Ok(())
}

/// Loads the router as Qwen3.5-35B-A3B stores it, full precision output_input. It reads the layer input before the
/// rotation and writes the f32 logits the top-k kernel takes.
async fn load_router<B: Backend>(
    num_experts: u32,
    model_dimension: u32,
    data_type: DataType,
    tree: &ParameterTree<'_, B>,
    context: &B::Context,
) -> Result<LinearMatmul<B>, MoeMlpNewError<B>> {
    let weights_tree = tree.subtree("weights");
    let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
    let AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
        layout: Layout::OutputInput,
        ..
    }) = spec
    else {
        return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
            "the router must be full precision output_input, got {spec:?}"
        )));
    };
    let router = LinearMatmul::load(
        model_dimension,
        num_experts,
        data_type,
        data_type,
        DataType::F32,
        spec,
        None,
        &weights_tree,
        None,
        context,
    )
    .await
    .map_err(LinearBlockError::from)?;
    Ok(router)
}

/// Loads the shared expert's gate, a single output over the input the routed experts rotate. Its own rotation must use
/// their input signs, so only its quantized matrix is loaded, to run on that input as it is.
async fn load_shared_gate<B: Backend>(
    model_dimension: u32,
    data_type: DataType,
    routed_input_signs: &[i32],
    tree: &ParameterTree<'_, B>,
    context: &B::Context,
) -> Result<LinearMatmul<B>, MoeMlpNewError<B>> {
    let weights_tree = tree.subtree("weights");
    let (quantization_spec, _) = check_quantized_spec(
        "the shared expert gate",
        weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?,
        IncoherenceProcessingMode::Input,
    )?;
    check_reads_routed_rotation(tree, routed_input_signs, "shared expert gate").await?;
    let gate = LinearMatmul::load(
        model_dimension,
        /* output_dim */ 1,
        data_type,
        data_type,
        data_type,
        quantization_spec,
        None,
        &weights_tree.subtree("quantized"),
        None,
        context,
    )
    .await
    .map_err(LinearBlockError::from)?;
    Ok(gate)
}

/// One projection of all routed experts, stored as `[num_experts, ...]` planes and rotated on both sides.
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
    /// Loads the projection and returns the input signs, which all experts share so that each input row is rotated
    /// once for all of its routes. `a_per_token` reads one input row per token instead of one per route.
    async fn load(
        num_experts: u32,
        input_dim: u32,
        output_dim: u32,
        data_type: DataType,
        a_per_token: bool,
        tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Box<[i32]>), MoeMlpNewError<B>> {
        let weights_tree = tree.subtree("weights");
        let RoutedFormat {
            quantization,
            bits,
            b_expert_stride_bytes,
        } = check_routed_projection(weights_tree.metadata("spec")?, num_experts, input_dim, output_dim)?;

        let input_signs =
            weights_tree.read_slice::<i32>("incoherence_signs.input_signs", &[num_experts, input_dim]).await?;
        if !experts_share_input_signs(&input_signs, input_dim as usize) {
            return Err(MoeMlpNewError::UnsupportedConfiguration(
                "routed experts must share their input signs".to_string(),
            ));
        }
        let output_hadamard_factors = weights_tree
            .read_buffer("incoherence_signs.output_signs", &[num_experts, output_dim], DataType::I32)
            .await?;

        let quantized_tree = weights_tree.subtree("quantized");
        let storage_type = quantization.mode.storage_type();
        let packing_divisor = quantization.mode.packing_divisor();
        let params = QuantParams::new(QuantParamsLayout::GroupOutput, output_dim, input_dim / MOE_GEMV_GROUP_SIZE);
        let weights = quantized_tree
            .read_buffer("weights", &[num_experts, output_dim, input_dim / packing_divisor], storage_type)
            .await?;
        let [scale_rows, scale_columns] = params.scale_shape();
        let scales = quantized_tree.read_buffer("scales", &[num_experts, scale_rows, scale_columns], data_type).await?;
        let (zero_points, zero_point_expert_stride, zero_point_strides) =
            if quantization.method == QuantizationMethod::ScaleZeroPoint {
                let [zero_point_rows, zero_point_columns] = params.zero_point_shape(quantization.mode);
                let zero_points = quantized_tree
                    .read_buffer("zero_points", &[num_experts, zero_point_rows, zero_point_columns], storage_type)
                    .await?;
                (Some(zero_points), zero_point_rows * zero_point_columns, params.zero_point_strides(quantization.mode))
            } else {
                (None, 0, QuantParamsStrides::default())
            };

        let kernel =
            <B::Kernels as Kernels>::MoeGemvKernel::new(context, data_type, MOE_GEMV_GROUP_SIZE, bits, a_per_token)
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
            input_signs[..input_dim as usize].into(),
        ))
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        expert_ids: impl BufferRef<Backend = B>,
        num_routes: u32,
        active_experts_per_token: u32,
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
            active_experts_per_token,
            self.output_dim / MOE_GEMV_OUTPUT_ROW_TILE,
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

pub struct MoeMlp<B: Backend> {
    router: LinearMatmul<B>,
    router_top_k: <B::Kernels as Kernels>::MoeRouterTopKKernel,
    input_rotation: ActivationTransform<B>,
    input_signs: B::GlobalGpuBuffer,
    routed_up: RoutedProjection<B>,
    routed_activation: MlpGateActMulEncodable<B>,
    routed_down: RoutedProjection<B>,
    shared_expert: DenseMlp<B>,
    shared_gate: LinearMatmul<B>,
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
        check_config(config, data_type)?;
        let num_experts = config.num_routed_experts;
        let hidden_dimension = config.expert_hidden_dim;
        let expert_config = &config.expert_config;

        let router =
            load_router(num_experts, model_dimension, data_type, &parameter_tree.subtree("router"), context).await?;
        let router_top_k =
            <B::Kernels as Kernels>::MoeRouterTopKKernel::new(context, DataType::F32, MOE_ROUTER_MAX_TOP_K)
                .map_err(MoeMlpNewError::Backend)?;

        // The up projection reads one input row per token, the down projection one row per route.
        let routed_tree = parameter_tree.subtree("routed_experts");
        let (routed_up, input_signs) = RoutedProjection::load(
            num_experts,
            model_dimension,
            2 * hidden_dimension,
            data_type,
            /* a_per_token */ true,
            &routed_tree.subtree("up_projection"),
            context,
        )
        .await?;
        let (routed_down, hidden_signs) = RoutedProjection::load(
            num_experts,
            hidden_dimension,
            model_dimension,
            data_type,
            /* a_per_token */ false,
            &routed_tree.subtree("down_projection"),
            context,
        )
        .await?;
        let routed_activation = MlpGateActMulEncodable::new(
            hidden_dimension,
            data_type,
            expert_config.activation.clone(),
            expert_config.gate_clipping,
            expert_config.up_clipping,
            Some(LinearInputPreparation {
                rht_signs: sign_buffer(&hidden_signs, context)?,
                activation_format: ActivationFormat::Bf16,
            }),
            context,
        )
        .map_err(MoeMlpNewError::Backend)?;

        let shared_tree = parameter_tree.subtree("shared_experts").unit_mixture();
        check_reads_routed_rotation(&shared_tree.subtree("up_projection"), &input_signs, "shared expert").await?;
        let shared_expert = DenseMlp::new_self_contained(
            model_dimension,
            hidden_dimension,
            data_type,
            expert_config,
            &shared_tree,
            context,
        )
        .await?;
        let shared_gate =
            load_shared_gate(model_dimension, data_type, &input_signs, &parameter_tree.subtree("gate"), context)
                .await?;

        let combine = <B::Kernels as Kernels>::MoeCombineKernel::new(context, data_type, MOE_COMBINE_BLOCK_DIM)
            .map_err(MoeMlpNewError::Backend)?;

        Ok(Self {
            router,
            router_top_k,
            input_rotation: ActivationTransform::input_rht(context, data_type, /* in_place */ true)
                .map_err(MoeMlpNewError::Backend)?,
            input_signs: sign_buffer(&input_signs, context)?,
            routed_up,
            routed_activation,
            routed_down,
            shared_expert,
            shared_gate,
            combine,
            model_dimension,
            num_routed_experts: num_experts,
            num_active_routed_experts: config.num_active_routed_experts,
            data_type,
        })
    }
}

impl<B: Backend> Mlp<B> for MoeMlp<B> {
    fn encode(
        &self,
        mut input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (moe)");
        let top_k = self.num_active_routed_experts;
        let num_routes = batch_dim * top_k;

        let router_logits = self.router.encode_ref(&input, batch_dim, command_buffer)?;
        let mut expert_ids = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, top_k], DataType::U32)
            .map_err(MlpEncodeError::Backend)?;
        let mut expert_weights = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, top_k], DataType::F32)
            .map_err(MlpEncodeError::Backend)?;
        self.router_top_k.encode(
            &router_logits,
            &mut expert_ids,
            &mut expert_weights,
            self.num_routed_experts,
            top_k,
            batch_dim,
            command_buffer,
        );

        // A shared expert that quantizes its activations rotates them while quantizing, so it reads the input before
        // the rotation below.
        let quantized_shared_output =
            if self.shared_expert.select_up_activation_format(batch_dim, command_buffer.context())
                != ActivationFormat::Bf16
            {
                Some(self.shared_expert.encode_ref(&input, batch_dim, command_buffer)?)
            } else {
                None
            };

        // One rotation serves the routed experts, the bf16 shared expert and the gate.
        self.input_rotation.encode_fp_in_place(
            &mut input,
            &self.input_signs,
            None,
            batch_dim,
            self.model_dimension,
            command_buffer,
        );
        let fused_up = self
            .routed_up
            .encode(&input, &expert_ids, num_routes, top_k, command_buffer)
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
            unreachable!("a bf16 activation is full precision");
        };
        let routed_outputs = self
            .routed_down
            .encode(&down_input, &expert_ids, num_routes, top_k, command_buffer)
            .map_err(MlpEncodeError::Backend)?;

        let shared_output = match quantized_shared_output {
            Some(shared_output) => shared_output,
            None => self.shared_expert.encode_prerotated_bf16(&input, batch_dim, command_buffer)?,
        };
        let shared_gate = self.shared_gate.encode_ref(&input, batch_dim, command_buffer)?;

        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], self.data_type)
            .map_err(MlpEncodeError::Backend)?;
        self.combine.encode(
            &routed_outputs,
            &expert_weights,
            &shared_output,
            &shared_gate,
            &mut output,
            self.model_dimension,
            top_k,
            batch_dim,
            command_buffer,
        );

        Ok(output)
    }
}
