use derive_more::Debug;
use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context,
        gpu_types::{
            gemm::{GemmBLayout, GemmBPrologueKind, GemmDTransform},
            hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
            quantization_method::QuantizationMethod,
        },
        kernel::{
            ActivationTransform, Kernels, MoeCombineKernel, MoeGemvKernel, MoeRouterTopKKernel,
            matmul::{ActivationFormat, QuantParams, QuantParamsLayout},
        },
    },
    config::{
        mlp::{mixture_of_experts::MixtureOfExpertsConfig, routing_function::AnyRoutingFunction},
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        },
    },
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError, LinearInput, LinearInputPreparation, LinearMatmul},
        mlp::{
            Mlp, MlpEncodeError,
            dense::{DenseMlp, DenseMlpNewError},
            gate_act_mul::MlpGateActMulEncodable,
        },
        weight_matrix::{ParsedWeightSpec, WeightMatrixError, parse_spec},
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

const MOE_GEMV_OUTPUT_ROW_TILE: u32 = 32;
const FP_K_BLOCK: u32 = 128;

pub(crate) fn verify_uniform_expert_signs(
    signs: &[i32],
    num_experts: usize,
    dim: usize,
) -> bool {
    if num_experts <= 1 || signs.len() < num_experts * dim {
        return true;
    }
    let row0 = &signs[..dim];
    for e in 1..num_experts {
        let row = &signs[e * dim..(e + 1) * dim];
        if row != row0 {
            return false;
        }
    }
    true
}

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

struct RoutedProjection<B: Backend> {
    kernel: <B::Kernels as Kernels>::MoeGemvKernel,
    input_rht: Option<(ActivationTransform<B>, B::GlobalBuffer)>,
    weights: B::GlobalBuffer,
    scales: Option<B::GlobalBuffer>,
    zero_points: Option<B::GlobalBuffer>,
    biases: Option<B::GlobalBuffer>,
    shared_weights: Option<B::GlobalBuffer>,
    shared_scales: Option<B::GlobalBuffer>,
    shared_zero_points: Option<B::GlobalBuffer>,
    shared_biases: Option<B::GlobalBuffer>,
    output_bias: Option<B::GlobalBuffer>,
    output_hadamard_factors: Option<B::GlobalBuffer>,
    shared_output_bias: Option<B::GlobalBuffer>,
    shared_output_hadamard_factors: Option<B::GlobalBuffer>,
    input_dim: u32,
    output_dim: u32,
    data_type: DataType,
    has_shared: bool,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
    zero_point_output_stride: u32,
    zero_point_group_stride: u32,
}

impl<B: Backend> RoutedProjection<B> {
    fn load(
        num_experts: u32,
        input_dim: u32,
        output_dim: u32,
        data_type: DataType,
        has_biases: bool,
        fuse_input_rht_externally: bool,
        proj_tree: &ParameterTree<B>,
        shared_proj_tree: Option<&ParameterTree<B>>,
        context: &B::Context,
    ) -> Result<(Self, Option<LinearInputPreparation<B>>), MoeMlpNewError<B>> {
        if !matches!(data_type, DataType::BF16 | DataType::F32) {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "unsupported data type for MoE routed projection: {data_type:?}"
            )));
        }

        let weights_tree = proj_tree.subtree("weights");
        let outer_spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

        let (quant_tree, inner_spec, input_signs, output_hadamard_factors) = match outer_spec {
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                incoherence_block_size: Some(block_size),
                incoherence_processing_mode:
                    mode @ (IncoherenceProcessingMode::Input | IncoherenceProcessingMode::InputOutput),
                ..
            }) if block_size == HADAMARD_TRANSFORM_BLOCK_SIZE => {
                let signs = match weights_tree.read_buffer(
                    "incoherence_signs.input_signs",
                    &[num_experts, input_dim],
                    DataType::I32,
                ) {
                    Ok(buf) => {
                        let signs_slice = (&buf).as_slice::<i32>();
                        if !verify_uniform_expert_signs(signs_slice, num_experts as usize, input_dim as usize) {
                            return Err(MoeMlpNewError::UnsupportedConfiguration(
                                "incoherence input_signs vary across experts in routed projection".to_string(),
                            ));
                        }
                        buf
                    },
                    Err(ParameterLoaderError::InvalidTensor {
                        ..
                    }) => weights_tree.read_buffer("incoherence_signs.input_signs", &[input_dim], DataType::I32)?,
                    Err(err) => return Err(err.into()),
                };
                let out_signs = (mode == IncoherenceProcessingMode::InputOutput)
                    .then(|| {
                        weights_tree.read_buffer(
                            "incoherence_signs.output_signs",
                            &[num_experts, output_dim],
                            DataType::I32,
                        )
                    })
                    .transpose()?;
                (weights_tree.subtree("quantized"), *quantization_spec, Some(signs), out_signs)
            },
            spec @ (AnyWeightMatrixSpec::FullPrecisionSpec(_)
            | AnyWeightMatrixSpec::IntSpec(_)
            | AnyWeightMatrixSpec::MLXSpec(_)) => (weights_tree, spec, None, None),
            spec => {
                return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                    "unsupported routed expert weight spec: {spec:?}"
                )));
            },
        };

        let ParsedWeightSpec {
            layout,
            quantization,
        } = parse_spec::<B>(&inner_spec)?;
        if layout != Layout::OutputInput {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "expected OutputInput layout for routed experts, got {layout:?}"
            )));
        }

        let output_bias =
            has_biases.then(|| proj_tree.read_buffer("biases", &[num_experts, output_dim], data_type)).transpose()?;

        let mut output_transform = GemmDTransform::empty();
        if output_hadamard_factors.is_some() {
            if !output_dim.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
                return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                    "output RHT requires output_dim ({output_dim}) to be a multiple of {HADAMARD_TRANSFORM_BLOCK_SIZE}"
                )));
            }
            output_transform |= GemmDTransform::RHT;
        }
        if output_bias.is_some() {
            output_transform |= GemmDTransform::BIAS;
        }

        let (
            weights,
            scales,
            zero_points,
            biases,
            b_prologue,
            bits,
            group_size,
            b_expert_stride_bytes,
            scale_expert_stride,
            zero_point_expert_stride,
            scale_strides,
            zero_point_strides,
        ) = match quantization {
            None => {
                let w = quant_tree.read_buffer("weights", &[num_experts, output_dim, input_dim], data_type)?;
                let stride_bytes = output_dim * input_dim * (data_type.size_in_bytes() as u32);
                (
                    w,
                    None,
                    None,
                    None,
                    GemmBPrologueKind::FullPrecision,
                    0u32,
                    0u32,
                    stride_bytes,
                    0u32,
                    0u32,
                    Default::default(),
                    Default::default(),
                )
            },
            Some(info) => {
                let bits = match info.mode.packing_divisor() {
                    2 => 4u32,
                    1 => 8u32,
                    _ => unreachable!(),
                };
                if !matches!(info.group_size, 32 | 64) {
                    return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                        "unsupported MoE quantization group size: {}",
                        info.group_size
                    )));
                }
                let packing_divisor = info.mode.packing_divisor();
                let storage_type = info.mode.storage_type();
                if !input_dim.is_multiple_of(packing_divisor) {
                    return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                        "input_dim {input_dim} is not divisible by packing_divisor {packing_divisor}"
                    )));
                }
                let groups = input_dim.div_ceil(info.group_size);
                let params = QuantParams::new(QuantParamsLayout::GroupOutput, output_dim, groups);
                let scale_shape = params.scale_shape();
                let scale_strides = params.scale_strides();
                let zp_strides = params.zero_point_strides(info.mode);

                let w = quant_tree.read_buffer(
                    "weights",
                    &[num_experts, output_dim, input_dim / packing_divisor],
                    storage_type,
                )?;
                let s = quant_tree.read_buffer("scales", &[num_experts, scale_shape[0], scale_shape[1]], data_type)?;
                let b_stride = output_dim * (input_dim / packing_divisor);
                let s_stride = scale_shape[0] * scale_shape[1];

                match info.method {
                    QuantizationMethod::ScaleZeroPoint => {
                        let zp_shape = params.zero_point_shape(info.mode);
                        let zp = quant_tree.read_buffer(
                            "zero_points",
                            &[num_experts, zp_shape[0], zp_shape[1]],
                            storage_type,
                        )?;
                        let zp_stride = zp_shape[0] * zp_shape[1];
                        (
                            w,
                            Some(s),
                            Some(zp),
                            None,
                            GemmBPrologueKind::ScaleZeroPointDequant,
                            bits,
                            info.group_size,
                            b_stride,
                            s_stride,
                            zp_stride,
                            scale_strides,
                            zp_strides,
                        )
                    },
                    QuantizationMethod::ScaleBias => {
                        let b = quant_tree.read_buffer(
                            "biases",
                            &[num_experts, scale_shape[0], scale_shape[1]],
                            data_type,
                        )?;
                        (
                            w,
                            Some(s),
                            None,
                            Some(b),
                            GemmBPrologueKind::ScaleBiasDequant,
                            bits,
                            info.group_size,
                            b_stride,
                            s_stride,
                            0u32,
                            scale_strides,
                            zp_strides,
                        )
                    },
                    QuantizationMethod::ScaleSymmetric => (
                        w,
                        Some(s),
                        None,
                        None,
                        GemmBPrologueKind::ScaleSymmetricDequant,
                        bits,
                        info.group_size,
                        b_stride,
                        s_stride,
                        0u32,
                        scale_strides,
                        zp_strides,
                    ),
                }
            },
        };

        let (
            shared_weights,
            shared_scales,
            shared_zero_points,
            shared_biases,
            shared_output_bias,
            shared_output_hadamard_factors,
        ) = if let Some(shared_tree) = shared_proj_tree {
            let shared_weights_tree = shared_tree.subtree("weights");
            let shared_quant_tree = shared_weights_tree.subtree("quantized");

            let shared_w = match quantization {
                None => shared_quant_tree.read_buffer("weights", &[1, output_dim, input_dim], data_type)?,
                Some(info) => {
                    let packing_divisor = info.mode.packing_divisor();
                    let storage_type = info.mode.storage_type();
                    shared_quant_tree.read_buffer(
                        "weights",
                        &[1, output_dim, input_dim / packing_divisor],
                        storage_type,
                    )?
                },
            };

            let (shared_s, shared_zp, shared_b) = match quantization {
                None => (None, None, None),
                Some(info) => {
                    let groups = input_dim.div_ceil(info.group_size);
                    let params = QuantParams::new(QuantParamsLayout::GroupOutput, output_dim, groups);
                    let scale_shape = params.scale_shape();
                    let s = shared_quant_tree.read_buffer("scales", &[1, scale_shape[0], scale_shape[1]], data_type)?;
                    match info.method {
                        QuantizationMethod::ScaleZeroPoint => {
                            let zp_shape = params.zero_point_shape(info.mode);
                            let zp = shared_quant_tree.read_buffer(
                                "zero_points",
                                &[1, zp_shape[0], zp_shape[1]],
                                info.mode.storage_type(),
                            )?;
                            (Some(s), Some(zp), None)
                        },
                        QuantizationMethod::ScaleBias => {
                            let b = shared_quant_tree.read_buffer(
                                "biases",
                                &[1, scale_shape[0], scale_shape[1]],
                                data_type,
                            )?;
                            (Some(s), None, Some(b))
                        },
                        QuantizationMethod::ScaleSymmetric => (Some(s), None, None),
                    }
                },
            };

            let shared_out_bias =
                has_biases.then(|| shared_tree.read_buffer("biases", &[1, output_dim], data_type)).transpose()?;

            let shared_out_hadamard = if output_hadamard_factors.is_some() {
                Some(shared_weights_tree.read_buffer(
                    "incoherence_signs.output_signs",
                    &[1, output_dim],
                    DataType::I32,
                )?)
            } else {
                None
            };

            if input_signs.is_some() {
                let _ = shared_weights_tree
                    .read_slice::<i32>("incoherence_signs.input_signs", &[1, input_dim])
                    .or_else(|_| shared_weights_tree.read_slice::<i32>("incoherence_signs.input_signs", &[input_dim]));
            }

            (Some(shared_w), shared_s, shared_zp, shared_b, shared_out_bias, shared_out_hadamard)
        } else {
            (None, None, None, None, None, None)
        };

        let block_size = if bits == 0 {
            FP_K_BLOCK
        } else if bits == 4 {
            512
        } else {
            256
        };
        let input_aligned = input_dim.is_multiple_of(block_size);
        let full_tile = output_dim.is_multiple_of(MOE_GEMV_OUTPUT_ROW_TILE);
        let a_per_token = !fuse_input_rht_externally;
        let has_shared = shared_weights.is_some();

        let kernel = <B::Kernels as Kernels>::MoeGemvKernel::new(
            context,
            data_type,
            group_size,
            bits,
            b_prologue,
            output_transform,
            input_aligned,
            false,
            GemmBLayout::RowMajor,
            full_tile,
            a_per_token,
            has_shared,
        )
        .map_err(MoeMlpNewError::Backend)?;

        let (input_rht, external_prep) = match input_signs {
            Some(signs) if fuse_input_rht_externally => (
                None,
                Some(LinearInputPreparation {
                    rht_signs: signs,
                    activation_quantization: None,
                }),
            ),
            Some(signs) => {
                let rht = ActivationTransform::input_rht(context, data_type, true).map_err(MoeMlpNewError::Backend)?;
                (Some((rht, signs)), None)
            },
            None => (None, None),
        };

        let has_zero_points = zero_points.is_some();
        Ok((
            Self {
                kernel,
                input_rht,
                weights,
                scales,
                zero_points,
                biases,
                shared_weights,
                shared_scales,
                shared_zero_points,
                shared_biases,
                output_bias,
                output_hadamard_factors,
                shared_output_bias,
                shared_output_hadamard_factors,
                input_dim,
                output_dim,
                data_type,
                has_shared,
                b_expert_stride_bytes,
                scale_expert_stride,
                zero_point_expert_stride,
                scale_output_stride: scale_strides.output_stride,
                scale_group_stride: scale_strides.group_stride,
                zero_point_output_stride: zp_strides_or_zero(has_zero_points, zero_point_strides.output_stride),
                zero_point_group_stride: zp_strides_or_zero(has_zero_points, zero_point_strides.group_stride),
            },
            external_prep,
        ))
    }

    fn rotate_input_in_place(
        &self,
        input: &mut B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        if let Some((rht, signs)) = &self.input_rht {
            rht.encode_fp_in_place(input, signs, None, batch_dim, self.input_dim, command_buffer);
        }
    }

    fn encode_prerotated(
        &self,
        input: impl BufferRef<Backend = B>,
        expert_ids: impl BufferRef<Backend = B>,
        batch_dim: u32,
        active_experts_per_token: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let num_routed_routes = batch_dim * active_experts_per_token;
        let num_routes = num_routed_routes
            + if self.has_shared {
                batch_dim
            } else {
                0
            };
        let mut output = command_buffer.allocate_scratch_for_shape(&[num_routes, self.output_dim], self.data_type)?;
        let group_count_x = self.output_dim.div_ceil(MOE_GEMV_OUTPUT_ROW_TILE);

        self.kernel.encode(
            &self.weights,
            self.scales.as_ref(),
            self.zero_points.as_ref(),
            self.biases.as_ref(),
            self.shared_weights.as_ref(),
            self.shared_scales.as_ref(),
            self.shared_zero_points.as_ref(),
            self.shared_biases.as_ref(),
            input,
            expert_ids,
            &mut output,
            self.output_bias.as_ref(),
            self.output_hadamard_factors.as_ref(),
            self.shared_output_bias.as_ref(),
            self.shared_output_hadamard_factors.as_ref(),
            self.input_dim,
            self.output_dim,
            num_routes,
            active_experts_per_token,
            group_count_x,
            self.b_expert_stride_bytes,
            self.scale_expert_stride,
            self.zero_point_expert_stride,
            self.scale_output_stride,
            self.scale_group_stride,
            self.zero_point_output_stride,
            self.zero_point_group_stride,
            num_routed_routes,
            command_buffer,
        );

        Ok(output)
    }
}

const fn zp_strides_or_zero(
    has_zp: bool,
    stride: u32,
) -> u32 {
    if has_zp {
        stride
    } else {
        0
    }
}

fn read_first_input_signs<B: Backend>(
    proj_tree: &ParameterTree<B>,
    num_experts: Option<u32>,
    input_dim: u32,
) -> Option<Vec<i32>> {
    let weights_tree = proj_tree.subtree("weights");
    let slice = match num_experts {
        Some(e) => weights_tree
            .read_slice::<i32>("incoherence_signs.input_signs", &[e, input_dim])
            .or_else(|_| weights_tree.read_slice::<i32>("incoherence_signs.input_signs", &[input_dim]))
            .ok()?,
        None => weights_tree.read_slice::<i32>("incoherence_signs.input_signs", &[input_dim]).ok()?,
    };
    Some(slice[..input_dim as usize].to_vec())
}

fn hadamard_transform(values: &mut [f32]) {
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

fn load_dequantized_unrotated_gate<B: Backend>(
    gate_tree: &ParameterTree<B>,
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
                .map_err(MoeMlpNewError::Parameter)?;
            let quant_tree = weights_tree.subtree("quantized");
            let packed_weights =
                quant_tree.read_slice::<u8>("weights", &[1, model_dimension / 2]).map_err(MoeMlpNewError::Parameter)?;
            let num_groups = model_dimension.div_ceil(64);
            let scales =
                quant_tree.read_slice::<bf16>("scales", &[num_groups, 4]).map_err(MoeMlpNewError::Parameter)?;
            let zero_points =
                quant_tree.read_slice::<u8>("zero_points", &[num_groups, 2]).map_err(MoeMlpNewError::Parameter)?;

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
                hadamard_transform(&mut stripe);
                for lane in 0..block_size {
                    let sign = input_signs[stripe_start + lane] as f32;
                    unrotated[stripe_start + lane] = bf16::from_f32(stripe[lane] * sign);
                }
            }
            Ok(Some(unrotated))
        },
        AnyWeightMatrixSpec::FullPrecisionSpec(_) => {
            let weights =
                weights_tree.read_slice::<bf16>("weights", &[1, model_dimension]).map_err(MoeMlpNewError::Parameter)?;
            Ok(Some(weights.to_vec()))
        },
        _ => Ok(None),
    }
}

pub struct MoeMlp<B: Backend> {
    router: Box<dyn Linear<B>>,
    router_top_k: <B::Kernels as Kernels>::MoeRouterTopKKernel,
    routed_up: RoutedProjection<B>,
    routed_gate: MlpGateActMulEncodable<B>,
    routed_down: RoutedProjection<B>,
    shared_expert: Option<DenseMlp<B>>,
    shared_gate: Option<Box<dyn Linear<B>>>,
    has_folded_gate: bool,
    shared_uses_routed_rht: bool,
    gate_uses_routed_rht: bool,
    combine: <B::Kernels as Kernels>::MoeCombineKernel,
    model_dimension: u32,
    num_routed_experts: u32,
    num_active_routed_experts: u32,
    data_type: DataType,
}

impl<B: Backend> MoeMlp<B> {
    pub fn new(
        model_dimension: u32,
        data_type: DataType,
        config: &MixtureOfExpertsConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, MoeMlpNewError<B>> {
        let AnyRoutingFunction::SoftmaxRouting(_) = config.routing_function;
        if config.num_active_routed_experts == 0 || config.num_active_routed_experts > 32 {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "num_active_routed_experts ({}) must be in 1..=32",
                config.num_active_routed_experts
            )));
        }
        if config.num_routed_experts == 0 || config.num_routed_experts > 512 {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "num_routed_experts ({}) must be in 1..=512",
                config.num_routed_experts
            )));
        }
        if config.num_shared_experts > 1 {
            return Err(MoeMlpNewError::UnsupportedConfiguration(format!(
                "num_shared_experts ({}) > 1 is not yet supported",
                config.num_shared_experts
            )));
        }

        let gate_tree = parameter_tree.subtree("gate");
        let gate_row = if config.num_shared_experts == 1 && config.gate_config.is_some() && !config.router_has_biases {
            load_dequantized_unrotated_gate(&gate_tree, model_dimension)?
        } else {
            None
        };

        let router_tree = parameter_tree.subtree("router");
        let router_weights_tree = router_tree.subtree("weights");
        let router_spec = router_weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let folded_router =
            if let (Some(gate_row), AnyWeightMatrixSpec::FullPrecisionSpec(_)) = (&gate_row, &router_spec) {
                let router_slice = router_weights_tree
                    .read_slice::<bf16>("weights", &[config.num_routed_experts, model_dimension])
                    .map_err(MoeMlpNewError::Parameter)?;
                let total_rows = config.num_routed_experts + 1;
                let mut folded_weights = Vec::with_capacity((total_rows * model_dimension) as usize);
                folded_weights.extend_from_slice(&router_slice);
                folded_weights.extend_from_slice(gate_row);
                let mut buffer =
                    context.create_buffer(folded_weights.len() * size_of::<bf16>()).map_err(MoeMlpNewError::Backend)?;
                buffer.copyin(&folded_weights);
                let router_matmul = LinearMatmul::new_full_precision(
                    buffer,
                    model_dimension,
                    total_rows,
                    data_type,
                    data_type,
                    DataType::F32,
                    context,
                )
                .map_err(|e| MoeMlpNewError::Linear(e.into()))?;
                Some(Box::new(router_matmul) as Box<dyn Linear<B>>)
            } else {
                None
            };

        let has_folded_gate = folded_router.is_some();
        let router = match folded_router {
            Some(r) => r,
            None => <dyn Linear<B>>::new_mixed_precision(
                model_dimension,
                config.num_routed_experts,
                data_type,
                data_type,
                DataType::F32,
                config.router_has_biases,
                &router_tree,
                context,
            )?,
        };
        let max_top_k = if config.num_active_routed_experts <= 8 {
            8
        } else if config.num_active_routed_experts <= 16 {
            16
        } else {
            32
        };
        let router_top_k = <B::Kernels as Kernels>::MoeRouterTopKKernel::new(context, DataType::F32, max_top_k, false)
            .map_err(MoeMlpNewError::Backend)?;

        let routed_tree = parameter_tree.subtree("routed_experts");
        let routed_up_tree = routed_tree.subtree("up_projection");
        let routed_up_signs = read_first_input_signs(&routed_up_tree, Some(config.num_routed_experts), model_dimension);
        let routed_down_tree = routed_tree.subtree("down_projection");
        let routed_down_signs =
            read_first_input_signs(&routed_down_tree, Some(config.num_routed_experts), config.expert_hidden_dim);

        let shared_tree = parameter_tree.subtree("shared_experts").unit_mixture();
        let shared_up_signs = (config.num_shared_experts == 1)
            .then(|| read_first_input_signs(&shared_tree.subtree("up_projection"), None, model_dimension))
            .flatten();
        let shared_down_signs = (config.num_shared_experts == 1)
            .then(|| read_first_input_signs(&shared_tree.subtree("down_projection"), None, config.expert_hidden_dim))
            .flatten();

        let shared_uses_routed_rht = (routed_up_signs.is_none() && shared_up_signs.is_none()
            || (routed_up_signs.is_some() && routed_up_signs == shared_up_signs))
            && (routed_down_signs.is_none() && shared_down_signs.is_none()
                || (routed_down_signs.is_some() && routed_down_signs == shared_down_signs));

        let can_fold_shared = config.num_shared_experts == 1 && shared_uses_routed_rht;
        let shared_up_tree = can_fold_shared.then(|| shared_tree.subtree("up_projection"));
        let (routed_up, _) = RoutedProjection::load(
            config.num_routed_experts,
            model_dimension,
            2 * config.expert_hidden_dim,
            data_type,
            config.expert_config.has_up_biases,
            false,
            &routed_up_tree,
            shared_up_tree.as_ref(),
            context,
        )?;

        let shared_down_tree = can_fold_shared.then(|| shared_tree.subtree("down_projection"));
        let (routed_down, down_input_prep) = RoutedProjection::load(
            config.num_routed_experts,
            config.expert_hidden_dim,
            model_dimension,
            data_type,
            config.expert_config.has_down_biases,
            true,
            &routed_down_tree,
            shared_down_tree.as_ref(),
            context,
        )?;
        let routed_gate = MlpGateActMulEncodable::new(
            config.expert_hidden_dim,
            data_type,
            config.expert_config.activation.clone(),
            config.expert_config.gate_clipping,
            config.expert_config.up_clipping,
            down_input_prep,
            context,
        )
        .map_err(MoeMlpNewError::Backend)?;

        let shared_expert = (config.num_shared_experts == 1 && !routed_up.has_shared)
            .then(|| {
                DenseMlp::new_self_contained(
                    model_dimension,
                    config.expert_hidden_dim,
                    data_type,
                    &config.expert_config,
                    &shared_tree,
                    context,
                )
            })
            .transpose()?;

        let gate_tree = parameter_tree.subtree("gate");
        let gate_signs = (config.num_shared_experts > 0 && config.gate_config.is_some())
            .then(|| read_first_input_signs(&gate_tree, None, model_dimension))
            .flatten();
        let gate_uses_routed_rht = routed_up_signs.is_some() && routed_up_signs == gate_signs;
        let shared_gate = if has_folded_gate {
            None
        } else {
            (config.num_shared_experts > 0 && config.gate_config.is_some())
                .then(|| {
                    <dyn Linear<B>>::new_mixed_precision(
                        model_dimension,
                        1,
                        data_type,
                        data_type,
                        DataType::F32,
                        false,
                        &gate_tree,
                        context,
                    )
                })
                .transpose()?
        };

        let combine = <B::Kernels as Kernels>::MoeCombineKernel::new(
            context,
            data_type,
            routed_down.has_shared || shared_expert.is_some(),
            has_folded_gate || shared_gate.is_some(),
        )
        .map_err(MoeMlpNewError::Backend)?;

        Ok(Self {
            router,
            router_top_k,
            routed_up,
            routed_gate,
            routed_down,
            shared_expert,
            shared_gate,
            has_folded_gate,
            shared_uses_routed_rht,
            gate_uses_routed_rht,
            combine,
            model_dimension,
            num_routed_experts: config.num_routed_experts,
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

        let router_logits = self.router.encode_ref(&input, batch_dim, command_buffer)?;

        let mut expert_ids = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.num_active_routed_experts], DataType::U32)
            .map_err(MlpEncodeError::Backend)?;
        let mut expert_weights = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.num_active_routed_experts], DataType::F32)
            .map_err(MlpEncodeError::Backend)?;
        let logit_stride = if self.has_folded_gate {
            self.num_routed_experts + 1
        } else {
            self.num_routed_experts
        };
        self.router_top_k.encode(
            &router_logits,
            None::<&B::ScratchBuffer>,
            &mut expert_ids,
            &mut expert_weights,
            self.num_routed_experts,
            self.num_active_routed_experts,
            batch_dim,
            logit_stride,
            command_buffer,
        );

        // If shared_expert needs unrotated input (e.g. fused RHT + Int8 quantize during prefill),
        // encode it before rotating `input` in-place.
        let mut shared_output = match &self.shared_expert {
            Some(shared)
                if !self.shared_uses_routed_rht
                    || shared.select_up_activation_format(batch_dim, command_buffer.context())
                        != ActivationFormat::Bf16 =>
            {
                Some(shared.encode_ref(&input, batch_dim, command_buffer)?)
            },
            _ => None,
        };
        let mut shared_gate = match &self.shared_gate {
            Some(gate) if !self.gate_uses_routed_rht => Some(gate.encode_ref(&input, batch_dim, command_buffer)?),
            _ => None,
        };

        // Rotate `input` in-place once and reuse across `routed_up`, `shared_expert` (in BF16 mode), and `shared_gate`.
        self.routed_up.rotate_input_in_place(&mut input, batch_dim, command_buffer);

        let fused_up = self
            .routed_up
            .encode_prerotated(&input, &expert_ids, batch_dim, self.num_active_routed_experts, command_buffer)
            .map_err(MlpEncodeError::Backend)?;

        let num_routed_routes = batch_dim * self.num_active_routed_experts;
        let total_routes = num_routed_routes
            + if self.routed_up.has_shared {
                batch_dim
            } else {
                0
            };

        let down_input = self
            .routed_gate
            .encode_for_linear(&fused_up, total_routes, ActivationFormat::Bf16, command_buffer)
            .map_err(MlpEncodeError::Backend)?;
        drop(fused_up);
        let LinearInput::FullPrecision(down_input) = down_input else {
            unreachable!("ActivationFormat::Bf16 always produces FullPrecision LinearInput");
        };

        let routed_outputs = self
            .routed_down
            .encode_prerotated(&down_input, &expert_ids, batch_dim, self.num_active_routed_experts, command_buffer)
            .map_err(MlpEncodeError::Backend)?;
        drop(down_input);
        drop(expert_ids);

        if shared_output.is_none()
            && let Some(shared) = &self.shared_expert
        {
            shared_output = Some(shared.encode_prerotated_bf16(&input, batch_dim, command_buffer)?);
        }
        if !self.has_folded_gate
            && shared_gate.is_none()
            && let Some(gate) = &self.shared_gate
        {
            shared_gate = Some(gate.encode_prerotated_bf16(&input, batch_dim, command_buffer)?);
        }
        drop(input);

        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.model_dimension], self.data_type)
            .map_err(MlpEncodeError::Backend)?;

        let (gate_buf, gate_stride) = if self.has_folded_gate {
            let f32_size = size_of::<f32>();
            let gate_offset = (self.num_routed_experts as usize) * f32_size;
            let gate_slice = (&router_logits).subrange(gate_offset..);
            (Some(gate_slice), self.num_routed_experts + 1)
        } else {
            (shared_gate.as_ref().map(|g| g.subrange(0..)), 1)
        };

        if self.routed_down.has_shared {
            let elem_size = self.data_type.size_in_bytes();
            let routed_bytes = (num_routed_routes * self.model_dimension) as usize * elem_size;
            let total_bytes = (total_routes * self.model_dimension) as usize * elem_size;
            let shared_slice = (&routed_outputs).subrange(routed_bytes..total_bytes);
            self.combine.encode(
                &routed_outputs,
                &expert_weights,
                Some(shared_slice),
                gate_buf,
                &mut output,
                self.model_dimension,
                self.num_active_routed_experts,
                batch_dim,
                gate_stride,
                command_buffer,
            );
        } else {
            self.combine.encode(
                &routed_outputs,
                &expert_weights,
                shared_output.as_ref(),
                gate_buf,
                &mut output,
                self.model_dimension,
                self.num_active_routed_experts,
                batch_dim,
                gate_stride,
                command_buffer,
            );
        }

        drop(router_logits);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use futures::executor::block_on;
    use uzu_engine_macros::uzu_test;

    use super::*;
    use crate::{
        backends::{common::Context, cpu::Cpu},
        config::mlp::mixture_of_experts::MixtureOfExpertsConfig,
        safetensors::ParameterLoader,
    };

    fn make_test_moe_config(
        num_active: u32,
        num_routed: u32,
        num_shared: u32,
    ) -> MixtureOfExpertsConfig {
        serde_json::from_value(serde_json::json!({
            "type": "MixtureOfExpertsConfig",
            "expert_config": {
                "type": "DenseMLPConfig",
                "linear_config": {},
                "activation": {
                    "type": "SiLU",
                    "alpha": 1.0
                },
                "has_up_biases": false,
                "has_down_biases": false,
                "gate_clipping": null,
                "up_clipping": null
            },
            "router_config": {},
            "routing_function": {
                "type": "SoftmaxRouting"
            },
            "num_routed_experts": num_routed,
            "num_active_routed_experts": num_active,
            "router_has_biases": false,
            "num_shared_experts": num_shared,
            "expert_hidden_dim": 64,
            "gate_config": null
        }))
        .expect("valid MixtureOfExpertsConfig")
    }

    #[uzu_test]
    fn test_verify_uniform_expert_signs_identical_rows() {
        // 3 experts, dim 4, all rows identical
        let row = [1, -1, 1, 1];
        let signs: Vec<i32> = row.iter().copied().cycle().take(3 * 4).collect();
        assert!(verify_uniform_expert_signs(&signs, 3, 4));
    }

    #[uzu_test]
    fn test_verify_uniform_expert_signs_single_expert() {
        let row = [1, -1, 1, 1];
        assert!(verify_uniform_expert_signs(&row, 1, 4));
    }

    #[uzu_test]
    fn test_verify_uniform_expert_signs_empty_and_zero() {
        assert!(verify_uniform_expert_signs(&[], 0, 4));
        assert!(verify_uniform_expert_signs(&[], 1, 4));
        assert!(verify_uniform_expert_signs(&[1, -1], 1, 2));
    }

    #[uzu_test]
    fn test_verify_uniform_expert_signs_insufficient_length() {
        // If buffer is shorter than num_experts * dim, it gracefully returns true
        let signs = vec![1, -1, 1, 1];
        assert!(verify_uniform_expert_signs(&signs, 2, 4));
    }

    #[uzu_test]
    fn test_verify_uniform_expert_signs_flipped_sign_fails() {
        // 2 experts, dim 4, expert 1 has one flipped sign
        let mut signs = vec![
            1, -1, 1, 1, // expert 0
            1, -1, -1, 1, // expert 1 (flipped 3rd element)
        ];
        assert!(!verify_uniform_expert_signs(&signs, 2, 4));

        // Fix the flipped sign
        signs[6] = 1;
        assert!(verify_uniform_expert_signs(&signs, 2, 4));
    }

    #[uzu_test]
    fn test_verify_uniform_expert_signs_many_experts_and_corruption() {
        // 8 experts, dim 8, all identical
        let row = [1, -1, 1, -1, -1, 1, 1, -1];
        let mut signs: Vec<i32> = row.iter().copied().cycle().take(8 * 8).collect();
        assert!(verify_uniform_expert_signs(&signs, 8, 8));

        // Flip one sign in the 8th expert
        let last_idx = signs.len() - 1;
        signs[last_idx] = -signs[last_idx];
        assert!(!verify_uniform_expert_signs(&signs, 8, 8));
    }

    #[uzu_test]
    fn test_moe_mlp_config_validation_bounds() {
        let temp_path = std::env::temp_dir().join(format!(
            "moe_test_{}_{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let header_json = serde_json::json!({}).to_string();
        let header_bytes = header_json.as_bytes();
        let header_len = (header_bytes.len() as u64).to_le_bytes();

        {
            let mut file = std::fs::File::create(&temp_path).expect("failed to create temp file");
            file.write_all(&header_len).unwrap();
            file.write_all(header_bytes).unwrap();
            file.flush().unwrap();
        }

        let std_file = std::fs::File::open(&temp_path).expect("failed to open temp file");
        let context = block_on(<Cpu as Backend>::Context::new()).expect("failed to create Cpu context");
        let loader = ParameterLoader::<Cpu>::new(&std_file, context.as_ref()).expect("failed to create loader");
        let tree = loader.tree();

        // 1. num_active_routed_experts == 0 fails
        let config = make_test_moe_config(0, 4, 0);
        let res = MoeMlp::<Cpu>::new(64, DataType::BF16, &config, &tree, context.as_ref());
        assert!(
            matches!(res, Err(MoeMlpNewError::UnsupportedConfiguration(msg)) if msg.contains("num_active_routed_experts (0)"))
        );

        // 2. num_active_routed_experts > 32 fails
        let config = make_test_moe_config(33, 64, 0);
        let res = MoeMlp::<Cpu>::new(64, DataType::BF16, &config, &tree, context.as_ref());
        assert!(
            matches!(res, Err(MoeMlpNewError::UnsupportedConfiguration(msg)) if msg.contains("num_active_routed_experts (33)"))
        );

        // 3. num_routed_experts == 0 fails
        let config = make_test_moe_config(4, 0, 0);
        let res = MoeMlp::<Cpu>::new(64, DataType::BF16, &config, &tree, context.as_ref());
        assert!(
            matches!(res, Err(MoeMlpNewError::UnsupportedConfiguration(msg)) if msg.contains("num_routed_experts (0)"))
        );

        // 4. num_routed_experts > 512 fails
        let config = make_test_moe_config(4, 513, 0);
        let res = MoeMlp::<Cpu>::new(64, DataType::BF16, &config, &tree, context.as_ref());
        assert!(
            matches!(res, Err(MoeMlpNewError::UnsupportedConfiguration(msg)) if msg.contains("num_routed_experts (513)"))
        );

        // 5. num_shared_experts > 1 fails
        let config = make_test_moe_config(4, 8, 2);
        let res = MoeMlp::<Cpu>::new(64, DataType::BF16, &config, &tree, context.as_ref());
        assert!(
            matches!(res, Err(MoeMlpNewError::UnsupportedConfiguration(msg)) if msg.contains("num_shared_experts (2)"))
        );

        let _ = std::fs::remove_file(temp_path);
    }
}
