//! Mixture-of-experts MLP block.

use std::mem::size_of_val;

use derive_more::Debug;
use half::bf16;
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
        linear::{Gather, InputRht, LinearBlockError, LinearInput, LinearInputPreparation, LinearMatmul},
        mlp::{
            Mlp, MlpEncodeError,
            dense::{DenseMlp, DenseMlpNewError},
            gate_act_mul::MlpGateActMulEncodable,
        },
        weight_matrix::{ParsedWeightSpec, WeightMatrixError, parse_spec},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

// Limits of the MoE kernels, from backends/metal/kernel/moe/common.h, their VARIANTS lists and the contract above
// MoeGemv: the router scores up to METAL_SIMD_SIZE x MOE_ROUTER_NUM_SIMDGROUPS experts and picks up to MAX_TOP_K, the
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

fn parse_quantized_spec<B: Backend>(
    weights_name: &str,
    spec: AnyWeightMatrixSpec,
    mode: IncoherenceProcessingMode,
) -> Result<AnyWeightMatrixSpec, MoeMlpNewError<B>> {
    match spec {
        AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            quantization_spec,
            incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
            incoherence_processing_mode,
            ..
        }) if incoherence_processing_mode == mode => Ok(*quantization_spec),
        spec => Err(MoeMlpNewError::UnsupportedConfiguration(format!(
            "{weights_name} must rotate {mode:?} in blocks of {HADAMARD_TRANSFORM_BLOCK_SIZE}, got {spec:?}"
        ))),
    }
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
    let quantization_spec = parse_quantized_spec(
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

fn hadamard_transform_slice(values: &mut [f32]) {
    let mut stride = 1;
    while stride < values.len() {
        for lane in 0..values.len() {
            if lane & stride == 0 {
                let a = values[lane];
                let b = values[lane | stride];
                values[lane] = a + b;
                values[lane | stride] = a - b;
            }
        }
        stride <<= 1;
    }
    let scale = 1.0 / (values.len() as f32).sqrt();
    for v in values.iter_mut() {
        *v *= scale;
    }
}

async fn load_dequantized_unrotated_gate<B: Backend>(
    gate_tree: &ParameterTree<'_, B>,
    model_dimension: u32,
) -> Result<Option<Vec<bf16>>, MoeMlpNewError<B>> {
    let weights_tree = gate_tree.subtree("weights");
    let Ok(spec) = weights_tree.metadata::<AnyWeightMatrixSpec>("spec") else {
        return Ok(None);
    };

    match spec {
        AnyWeightMatrixSpec::HybridSpec(hybrid) => {
            if hybrid.incoherence_block_size != Some(HADAMARD_TRANSFORM_BLOCK_SIZE)
                || hybrid.incoherence_processing_mode != IncoherenceProcessingMode::Input
            {
                return Ok(None);
            }
            let AnyWeightMatrixSpec::IntSpec(int_spec) = *hybrid.quantization_spec else {
                return Ok(None);
            };
            if int_spec.bits != 4 || int_spec.group_size != 64 {
                return Ok(None);
            }

            let input_signs = weights_tree
                .read_slice::<i32>("incoherence_signs.input_signs", &[model_dimension])
                .await
                .map_err(MoeMlpNewError::Parameter)?;
            let quant_tree = weights_tree.subtree("quantized");
            let packed_weights = quant_tree
                .read_slice::<u8>("weights", &[1, model_dimension / 2])
                .await
                .map_err(MoeMlpNewError::Parameter)?;
            let num_groups = model_dimension.div_ceil(64);
            let scales =
                quant_tree.read_slice::<bf16>("scales", &[num_groups, 4]).await.map_err(MoeMlpNewError::Parameter)?;
            let zero_points = quant_tree
                .read_slice::<u8>("zero_points", &[num_groups, 2])
                .await
                .map_err(MoeMlpNewError::Parameter)?;

            let mut dequantized = vec![0.0f32; model_dimension as usize];
            for (k, val) in dequantized.iter_mut().enumerate() {
                let byte_idx = k / 2;
                let b_val = packed_weights[byte_idx];
                let code = if k % 2 == 0 {
                    b_val & 0x0F
                } else {
                    (b_val >> 4) & 0x0F
                };
                let g = k / 64;
                let scale = scales[g * 4].to_f32();
                let zp_byte = zero_points[g * 2];
                let zp = (zp_byte & 0x0F) as f32;
                *val = scale * (code as f32 - zp);
            }

            let mut unrotated = vec![bf16::ZERO; model_dimension as usize];
            let block_size = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
            for stripe_start in (0..model_dimension as usize).step_by(block_size) {
                let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize];
                stripe.copy_from_slice(&dequantized[stripe_start..stripe_start + block_size]);
                hadamard_transform_slice(&mut stripe);
                for lane in 0..block_size {
                    let sign = input_signs[stripe_start + lane] as f32;
                    unrotated[stripe_start + lane] = bf16::from_f32(stripe[lane] * sign);
                }
            }
            Ok(Some(unrotated))
        },
        AnyWeightMatrixSpec::FullPrecisionSpec(_) => {
            let weights = weights_tree
                .read_slice::<bf16>("weights", &[1, model_dimension])
                .await
                .map_err(MoeMlpNewError::Parameter)?;
            Ok(Some(weights.to_vec()))
        },
        _ => Ok(None),
    }
}

/// One projection of all routed experts, stored as `[num_experts, ...]` planes and rotated on both sides.
struct RoutedProjection<B: Backend> {
    kernel: <B::Kernels as Kernels>::MoeGemvKernel,
    weights: B::GlobalGpuBuffer,
    scales: B::GlobalGpuBuffer,
    zero_points: Option<B::GlobalGpuBuffer>,
    shared_weights: Option<B::GlobalGpuBuffer>,
    shared_scales: Option<B::GlobalGpuBuffer>,
    shared_zero_points: Option<B::GlobalGpuBuffer>,
    output_hadamard_factors: B::GlobalGpuBuffer,
    shared_output_hadamard_factors: Option<B::GlobalGpuBuffer>,
    input_dim: u32,
    output_dim: u32,
    data_type: DataType,
    has_shared: bool,
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
        shared_tree: Option<&ParameterTree<'_, B>>,
        context: &B::Context,
    ) -> Result<(Self, Box<[i32]>), MoeMlpNewError<B>> {
        let weights_tree = tree.subtree("weights");
        let spec = weights_tree.metadata("spec")?;
        let quantization_spec = parse_quantized_spec("routed experts", spec, IncoherenceProcessingMode::InputOutput)?;
        let ParsedWeightSpec {
            layout,
            quantization,
        } = parse_spec::<B>(&quantization_spec)?;
        let Some(quantization) = quantization
            .filter(|quantization| layout == Layout::OutputInput && quantization.group_size == MOE_GEMV_GROUP_SIZE)
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "routed experts must be quantized output_input in groups of {MOE_GEMV_GROUP_SIZE}, got {quantization_spec:?}"
            )));
        };
        match (quantization.mode, quantization.method) {
            (QuantizationMode::U4, QuantizationMethod::ScaleZeroPoint)
            | (QuantizationMode::U8, QuantizationMethod::ScaleSymmetric) => {},
            (mode, method) => {
                return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                    "routed experts must be 4-bit with zero points or 8-bit symmetric, got {mode} {method}"
                )));
            },
        }
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
        let Some(b_expert_stride_bytes) = output_dim
            .checked_mul(input_dim / quantization.mode.packing_divisor())
            .filter(|plane_bytes| (num_experts - 1).checked_mul(*plane_bytes).is_some())
        else {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "{num_experts} expert planes of {output_dim} x {input_dim} {bits}-bit weights exceed 32-bit offsets"
            )));
        };

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

        let has_shared = shared_tree.is_some();
        let (shared_weights, shared_scales, shared_zero_points, shared_output_hadamard_factors) = if let Some(st) =
            shared_tree
        {
            let st_weights = st.subtree("weights");
            let st_quant = st_weights.subtree("quantized");
            let sw = st_quant.read_buffer("weights", &[output_dim, input_dim / packing_divisor], storage_type).await?;
            let ss = st_quant.read_buffer("scales", &[scale_rows, scale_columns], data_type).await?;
            let szp = if quantization.method == QuantizationMethod::ScaleZeroPoint {
                let [zero_point_rows, zero_point_columns] = params.zero_point_shape(quantization.mode);
                Some(st_quant.read_buffer("zero_points", &[zero_point_rows, zero_point_columns], storage_type).await?)
            } else {
                None
            };
            let sohf = st_weights.read_buffer("incoherence_signs.output_signs", &[output_dim], DataType::I32).await?;
            (Some(sw), Some(ss), szp, Some(sohf))
        } else {
            (None, None, None, None)
        };

        let kernel = <B::Kernels as Kernels>::MoeGemvKernel::new(
            context,
            data_type,
            MOE_GEMV_GROUP_SIZE,
            bits,
            a_per_token,
            has_shared,
        )
        .map_err(MoeMlpNewError::Backend)?;

        Ok((
            Self {
                kernel,
                weights,
                scales,
                zero_points,
                shared_weights,
                shared_scales,
                shared_zero_points,
                output_hadamard_factors,
                shared_output_hadamard_factors,
                input_dim,
                output_dim,
                data_type,
                has_shared,
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
        num_routed_routes: u32,
        active_experts_per_token: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = command_buffer.allocate_scratch_for_shape(&[num_routes, self.output_dim], self.data_type)?;
        self.kernel.encode(
            &self.weights,
            &self.scales,
            self.zero_points.as_ref(),
            self.shared_weights.as_ref(),
            self.shared_scales.as_ref(),
            self.shared_zero_points.as_ref(),
            input,
            expert_ids,
            &mut output,
            &self.output_hadamard_factors,
            self.shared_output_hadamard_factors.as_ref(),
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
            num_routed_routes,
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
    shared_input_rht: Option<InputRht<B>>,
    shared_gate: Option<LinearMatmul<B>>,
    has_folded_gate: bool,
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

        let gate_tree = parameter_tree.subtree("gate");
        let gate_row = load_dequantized_unrotated_gate(&gate_tree, model_dimension).await?;

        let router_tree = parameter_tree.subtree("router");
        let router_weights_tree = router_tree.subtree("weights");
        let router_spec = router_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

        let (router, has_folded_gate) =
            if let (Some(gate_row), AnyWeightMatrixSpec::FullPrecisionSpec(_)) = (&gate_row, &router_spec) {
                let router_slice = router_weights_tree
                    .read_slice::<bf16>("weights", &[num_experts, model_dimension])
                    .await
                    .map_err(MoeMlpNewError::Parameter)?;
                let total_rows = num_experts + 1;
                let mut folded_weights = Vec::with_capacity((total_rows * model_dimension) as usize);
                folded_weights.extend_from_slice(&router_slice);
                folded_weights.extend_from_slice(gate_row);
                let mut buffer = context
                    .create_buffer_cpu_gpu(folded_weights.len() * std::mem::size_of::<bf16>())
                    .map_err(MoeMlpNewError::Backend)?;
                buffer.copyin(&folded_weights);
                let router_matmul = LinearMatmul::new_full_precision(
                    buffer.handoff(),
                    model_dimension,
                    total_rows,
                    data_type,
                    data_type,
                    DataType::F32,
                    context,
                )
                .map_err(|e| MoeMlpNewError::Linear(e.into()))?;
                (router_matmul, true)
            } else {
                let r = load_router(num_experts, model_dimension, data_type, &router_tree, context).await?;
                (r, false)
            };
        let router_top_k =
            <B::Kernels as Kernels>::MoeRouterTopKKernel::new(context, DataType::F32, MOE_ROUTER_MAX_TOP_K)
                .map_err(MoeMlpNewError::Backend)?;

        // The up projection reads one input row per token, the down projection one row per route.
        let routed_tree = parameter_tree.subtree("routed_experts");
        let shared_tree = parameter_tree.subtree("shared_experts");
        let (routed_up, input_signs) = RoutedProjection::load(
            num_experts,
            model_dimension,
            2 * hidden_dimension,
            data_type,
            /* a_per_token */ true,
            &routed_tree.subtree("up_projection"),
            Some(&shared_tree.subtree("up_projection")),
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
            Some(&shared_tree.subtree("down_projection")),
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
        check_reads_routed_rotation(&shared_tree.subtree("up_projection"), &input_signs, "shared expert").await?;
        let (shared_expert, shared_preparation) = DenseMlp::new_for_fused_input(
            model_dimension,
            hidden_dimension,
            data_type,
            expert_config,
            &shared_tree,
            context,
        )
        .await?;
        let shared_input_rht = shared_preparation
            .map(|preparation| InputRht::new(data_type, preparation, false, context))
            .transpose()
            .map_err(MoeMlpNewError::Backend)?;
        let shared_gate = if has_folded_gate {
            None
        } else {
            Some(load_shared_gate(model_dimension, data_type, &input_signs, &gate_tree, context).await?)
        };

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
            shared_input_rht,
            shared_gate,
            has_folded_gate,
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

        let router_logits = self.router.encode_with_a(
            MatmulA::full_precision(&input),
            batch_dim,
            None::<Gather<&B::ScratchBuffer>>,
            command_buffer,
        )?;
        let mut expert_ids = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, top_k], DataType::U32)
            .map_err(MlpEncodeError::Backend)?;
        let mut expert_weights = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, top_k], DataType::F32)
            .map_err(MlpEncodeError::Backend)?;
        let logit_stride = if self.has_folded_gate {
            self.num_routed_experts + 1
        } else {
            self.num_routed_experts
        };
        self.router_top_k.encode(
            &router_logits,
            &mut expert_ids,
            &mut expert_weights,
            self.num_routed_experts,
            top_k,
            batch_dim,
            logit_stride,
            command_buffer,
        );

        let quantized_shared_input = match &self.shared_input_rht {
            Some(input_rht) => {
                let format = self.shared_expert.select_activation_format(batch_dim, command_buffer.context());
                if format != ActivationFormat::Bf16 {
                    Some(
                        input_rht
                            .prepare(&input, batch_dim, format, command_buffer)
                            .map_err(MlpEncodeError::Backend)?,
                    )
                } else {
                    None
                }
            },
            None => None,
        };

        // One rotation serves the routed experts, the shared expert, and the gate.
        self.input_rotation.encode_fp_in_place(
            &mut input,
            &self.input_signs,
            None,
            batch_dim,
            self.model_dimension,
            command_buffer,
        );
        let num_routed_routes = num_routes;
        let total_routes = num_routed_routes
            + if self.routed_up.has_shared {
                batch_dim
            } else {
                0
            };

        let fused_up = self
            .routed_up
            .encode(&input, &expert_ids, total_routes, num_routed_routes, top_k, command_buffer)
            .map_err(MlpEncodeError::Backend)?;
        let down_input = self
            .routed_activation
            .encode_for_linear(&fused_up, total_routes, ActivationFormat::Bf16, command_buffer)
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
            .encode(&down_input, &expert_ids, total_routes, num_routed_routes, top_k, command_buffer)
            .map_err(MlpEncodeError::Backend)?;

        let fallback_gate;
        let (gate_buf, gate_stride) = if self.has_folded_gate {
            let f32_size = std::mem::size_of::<f32>();
            let gate_offset = (self.num_routed_experts as usize) * f32_size;
            let gate_slice = (&router_logits).subrange(gate_offset..);
            (gate_slice, self.num_routed_experts + 1)
        } else {
            fallback_gate = self.shared_gate.as_ref().unwrap().encode_with_a(
                MatmulA::full_precision(&input),
                batch_dim,
                None::<Gather<&B::ScratchBuffer>>,
                command_buffer,
            )?;
            (fallback_gate.subrange(0..), 1)
        };

        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], self.data_type)
            .map_err(MlpEncodeError::Backend)?;

        if self.routed_down.has_shared {
            let elem_size = self.data_type.size_in_bytes();
            let routed_bytes = (num_routed_routes * self.model_dimension) as usize * elem_size;
            let total_bytes = (total_routes * self.model_dimension) as usize * elem_size;
            let shared_slice = (&routed_outputs).subrange(routed_bytes..total_bytes);
            self.combine.encode(
                &routed_outputs,
                &expert_weights,
                shared_slice,
                gate_buf,
                &mut output,
                self.model_dimension,
                top_k,
                batch_dim,
                gate_stride,
                command_buffer,
            );
        } else {
            let shared_input = match quantized_shared_input {
                Some(shared_input) => shared_input,
                None => LinearInput::<B>::full_precision(input),
            };
            let shared_output = self.shared_expert.encode_input(shared_input, batch_dim, command_buffer)?;
            self.combine.encode(
                &routed_outputs,
                &expert_weights,
                &shared_output,
                gate_buf,
                &mut output,
                self.model_dimension,
                top_k,
                batch_dim,
                gate_stride,
                command_buffer,
            );
        }

        Ok(output)
    }
}
