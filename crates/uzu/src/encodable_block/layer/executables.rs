//! Layer executables - a single decoder layer with mixer, norms, and MLP.

#[cfg(feature = "tracing")]
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use super::MixerExecutables;
#[cfg(feature = "tracing")]
use crate::backends::common::kernel::TensorAddBiasKernel;
use crate::{
    DataType,
    array::ArrayContextExt,
    backends::common::{
        ActivationConfig, Backend, Encoder, Kernels,
        kernel::{ElementWiseMulStridedKernel, TensorAddScaleKernel},
    },
    config::{DecoderLayerConfig, DecoderLayerType, LinearConfig, MixerConfig, NormalizationConfig},
    encodable_block::{
        Activation, Attention, DeltaNetMixer, EncodingParameters, Linear, MambaMixer, Mlp, QKNorm, RMSNorm, Rope,
        ShortConvMixer, TensorAddSwap,
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

struct PleLayerComponents<B: Backend> {
    gate: Box<dyn Linear<B>>,
    projection: Box<dyn Linear<B>>,
    post_norm: RMSNorm<B>,
    activation: Activation<B>,
    mul_strided_kernel: <B::Kernels as Kernels>::ElementWiseMulStridedKernel,
    add_swap: TensorAddSwap<B>,
    dim: usize,
    total_dim: usize,
}

/// A single decoder layer with all its components.
pub struct LayerExecutables<B: Backend> {
    #[allow(dead_code)]
    pub layer_index: usize,
    #[cfg(feature = "tracing")]
    pub tensor_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    pub pre_attention_norm: RMSNorm<B>,
    pub(crate) mixer: MixerExecutables<B>,
    pub post_attention_norm: Option<RMSNorm<B>>,
    pub pre_mlp_norm: RMSNorm<B>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<RMSNorm<B>>,
    /// Explicit residual add for layers with PLE/scalar. These must operate on the full
    /// residual sum (input + attn + mlp), so the normally-deferred MLP residual add is
    /// performed here instead of in the next layer's pre_attention_norm.
    mlp_residual_add: Option<TensorAddSwap<B>>,
    ple: Option<PleLayerComponents<B>>,
    layer_scalar: f32,
    layer_scalar_kernel: Option<<B::Kernels as Kernels>::TensorAddScaleKernel>,
    /// Zero-bias buffer for scalar-only multiply via TensorAddScale: (input + 0) * scale
    layer_scalar_zero_bias: Option<B::Buffer>,
    model_dim: usize,
    /// Per-layer attention dimensions for buffer reshaping. May differ from the max
    /// dimensions used for buffer allocation.
    attention_num_heads: usize,
    attention_num_groups: usize,
    attention_head_dim: usize,
    /// When true, the KV cache update is skipped to avoid overwriting the source layer's cache.
    is_kv_shared_layer: bool,
}

impl<B: Backend> LayerExecutables<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        layer_config: &DecoderLayerConfig,
        layer_type: &DecoderLayerType,
        layer_index: usize,
        model_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        attention_scale: Option<f32>,
        decoder_layer_loader: &ParameterTree<B::Context>,
        rope: Option<Rc<Rope<B>>>,
        ple_dim: Option<usize>,
        ple_linear_config: Option<&LinearConfig>,
        ple_norm_config: Option<&NormalizationConfig>,
        num_layers: usize,
        is_kv_shared_layer: bool,
        previous_layer_did_explicit_residual: bool,
    ) -> Self {
        let intermediate_data_type: DataType = match &layer_config.mixer_config {
            MixerConfig::Attention(attention) => attention.qkv_projection_config.activation_precision().into(),
            MixerConfig::Mamba(mamba) => mamba.in_projection_config.activation_precision().into(),
            MixerConfig::ShortConv(short_conv) => short_conv.in_projection_config.activation_precision().into(),
            MixerConfig::DeltaNet(config) => config.in_proj_config.activation_precision().into(),
        };

        #[cfg(feature = "tracing")]
        let tensor_add = TensorAddBiasKernel::new(context, intermediate_data_type, false)
            .expect("Failed to create TensorAddBiasKernel kernel"); // TODO: this function return Result

        let fused_residual_add = layer_index > 0 && !previous_layer_did_explicit_residual;
        let pre_attention_norm = RMSNorm::new(
            context,
            intermediate_data_type,
            layer_config.pre_attention_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &decoder_layer_loader.subtree("pre_mixer_norm").unwrap(),
            Some(ArrayId::Shortcut),
            fused_residual_add,
        )
        .expect("Failed to create RMS norm kernel");

        let (attn_num_heads, attn_num_groups, attn_head_dim) = match &layer_config.mixer_config {
            MixerConfig::Attention(attention_config) => (
                attention_config.num_heads.unwrap_or(num_heads),
                attention_config.num_groups.unwrap_or(num_groups),
                attention_config.head_dim.unwrap_or(head_dim),
            ),
            _ => (num_heads, num_groups, head_dim),
        };

        let mixer = match &layer_config.mixer_config {
            MixerConfig::Attention(attention_config) => {
                let rope_block = rope.expect("RoPE encoder missing for attention layer");

                let layer_num_heads = attn_num_heads;
                let layer_num_groups = attn_num_groups;
                let layer_head_dim = attn_head_dim;

                let q_dim = layer_num_heads * layer_head_dim;
                let kv_dim = layer_num_groups * layer_head_dim;

                let qkv_projection = <dyn Linear<B>>::new(
                    &attention_config.qkv_projection_config,
                    attention_config.has_qkv_biases,
                    model_dim,
                    [q_dim, kv_dim, kv_dim],
                    context,
                    &decoder_layer_loader.subtree("mixer.qkv_projection").unwrap(),
                    ArrayId::Main,
                    ArrayId::QKV,
                )
                .expect("Failed to create qkv projection");

                let has_gate = attention_config.has_gate || attention_config.gate_projection_config.is_some();
                let gate_projection = if has_gate {
                    let gate_config = attention_config
                        .gate_projection_config
                        .as_ref()
                        .unwrap_or(&attention_config.qkv_projection_config);
                    Some(
                        <dyn Linear<B>>::new(
                            gate_config,
                            false,
                            model_dim,
                            [q_dim],
                            context,
                            &decoder_layer_loader.subtree("mixer.gate_projection").unwrap(),
                            ArrayId::Main,
                            ArrayId::Gate,
                        )
                        .expect("Failed to create gate projection"),
                    )
                } else {
                    None
                };

                let qk_norm = if attention_config.query_norm_config.is_some()
                    || attention_config.key_norm_config.is_some()
                    || attention_config.value_norm_config.is_some()
                {
                    match QKNorm::new(
                        context,
                        intermediate_data_type,
                        attention_config.query_norm_config.clone(),
                        attention_config.key_norm_config.clone(),
                        attention_config.value_norm_config.clone(),
                        ArrayId::QKV,
                        &decoder_layer_loader.subtree("mixer").unwrap(),
                        layer_num_heads,
                        layer_num_groups,
                        layer_head_dim,
                    ) {
                        Ok(qk_norm) => Some(qk_norm),
                        Err(e) => panic!("Failed to create QK norm kernel for layer {}: {:?}", layer_index, e),
                    }
                } else {
                    None
                };

                let out_projection = <dyn Linear<B>>::new(
                    &attention_config.out_projection_config,
                    attention_config.has_out_biases,
                    q_dim,
                    [model_dim],
                    context,
                    &decoder_layer_loader.subtree("mixer.out_projection").unwrap(),
                    ArrayId::AttentionOutput,
                    ArrayId::Main,
                )
                .expect("Failed to create out projection");

                let attention = Attention::new(
                    context,
                    intermediate_data_type,
                    layer_index,
                    attention_scale,
                    attention_config.has_sinks,
                    attention_config.is_causal.unwrap_or(true),
                    attention_config.sliding_window_size,
                    has_gate,
                )
                .expect("Failed to create AttentionWrapper kernel");

                MixerExecutables::Attention {
                    qkv_projection,
                    gate_projection,
                    qk_norm,
                    rope: rope_block,
                    attention,
                    out_projection,
                }
            },
            MixerConfig::Mamba(mamba_config) => {
                let mixer = MambaMixer::new(
                    context,
                    layer_type.clone(),
                    mamba_config.clone(),
                    layer_index,
                    model_dim,
                    num_heads,
                    head_dim,
                    num_groups,
                    decoder_layer_loader,
                );
                MixerExecutables::StateSpace {
                    mixer,
                }
            },
            MixerConfig::ShortConv(short_conv_config) => {
                let mixer = ShortConvMixer::new(
                    context,
                    layer_type.clone(),
                    short_conv_config.clone(),
                    layer_index,
                    model_dim,
                    decoder_layer_loader,
                );
                MixerExecutables::ShortConv {
                    mixer,
                }
            },
            MixerConfig::DeltaNet(delta_net_config) => {
                let mixer =
                    DeltaNetMixer::new(context, delta_net_config.clone(), layer_index, model_dim, decoder_layer_loader)
                        .expect("Failed to create DeltaNet mixer");

                MixerExecutables::DeltaNet {
                    mixer,
                }
            },
        };

        let post_attention_norm = if let Some(norm_config) = &layer_config.post_attention_norm_config {
            Some(
                RMSNorm::new(
                    context,
                    intermediate_data_type,
                    norm_config.clone(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &decoder_layer_loader.subtree("post_mixer_norm").unwrap(),
                    None,
                    false,
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
        };

        let pre_mlp_norm = RMSNorm::new(
            context,
            intermediate_data_type,
            layer_config.pre_mlp_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &decoder_layer_loader.subtree("pre_mlp_norm").unwrap(),
            Some(ArrayId::Shortcut),
            true,
        )
        .expect("Failed to create RMS norm kernel");

        let mlp = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            model_dim,
            hidden_dim,
            context,
            &decoder_layer_loader.subtree("mlp").unwrap(),
        )
        .expect("Failed to create mlp block");

        let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
            Some(
                RMSNorm::new(
                    context,
                    intermediate_data_type,
                    norm_config.clone(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &decoder_layer_loader.subtree("post_mlp_norm").unwrap(),
                    None,
                    false,
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
        };

        let ple = match (ple_dim, ple_linear_config, ple_norm_config) {
            (Some(ple_dim), Some(linear_config), Some(norm_config)) if ple_dim > 0 => {
                let ple_data_type: DataType = linear_config.activation_precision().into();

                let gate = <dyn Linear<B>>::new(
                    linear_config,
                    false,
                    model_dim,
                    [ple_dim],
                    context,
                    &decoder_layer_loader.subtree("ple_gate").unwrap(),
                    ArrayId::Main,
                    ArrayId::PleGate,
                )
                .expect("Failed to create PLE gate");

                let projection = <dyn Linear<B>>::new(
                    linear_config,
                    false,
                    ple_dim,
                    [model_dim],
                    context,
                    &decoder_layer_loader.subtree("ple_projection").unwrap(),
                    ArrayId::PleGate,
                    ArrayId::Main,
                )
                .expect("Failed to create PLE projection");

                let post_norm = RMSNorm::new(
                    context,
                    ple_data_type,
                    norm_config.clone(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &decoder_layer_loader.subtree("post_ple_norm").unwrap(),
                    None,
                    false,
                )
                .expect("Failed to create post-PLE norm");

                let activation =
                    Activation::new(context, ple_data_type, ActivationConfig::GELU, ArrayId::PleGate, ArrayId::PleGate)
                        .expect("Failed to create PLE GELU activation");

                let mul_strided_kernel =
                    <B::Kernels as Kernels>::ElementWiseMulStridedKernel::new(context, ple_data_type)
                        .expect("Failed to create ElementWiseMulStrided kernel");

                let ple_total_dim = num_layers * ple_dim;

                let add_swap = TensorAddSwap::new(context, ple_data_type, ArrayId::Shortcut, ArrayId::Main)
                    .expect("Failed to create PLE add-swap");

                Some(PleLayerComponents {
                    gate,
                    projection,
                    post_norm,
                    activation,
                    mul_strided_kernel,
                    add_swap,
                    dim: ple_dim,
                    total_dim: ple_total_dim,
                })
            },
            _ => None,
        };

        let has_ple = ple.is_some();
        let mlp_residual_add = if has_ple || layer_config.has_layer_scalar {
            Some(
                TensorAddSwap::new(context, intermediate_data_type, ArrayId::Shortcut, ArrayId::Main)
                    .expect("Failed to create MLP residual add-swap"),
            )
        } else {
            None
        };

        let (layer_scalar, layer_scalar_kernel, layer_scalar_zero_bias) = if layer_config.has_layer_scalar {
            let scalar_array =
                decoder_layer_loader.leaf_array("layer_scalar").expect("Failed to load layer_scalar weight");
            let scalar_value = match scalar_array.data_type() {
                DataType::BF16 => {
                    let bytes = scalar_array.as_bytes();
                    let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
                    half::bf16::from_bits(raw).to_f32()
                },
                DataType::F16 => {
                    let bytes = scalar_array.as_bytes();
                    let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
                    half::f16::from_bits(raw).to_f32()
                },
                DataType::F32 => {
                    let bytes = scalar_array.as_bytes();
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                },
                dt => panic!("Failed to decode layer_scalar weight: unsupported data type {dt:?}"),
            };

            if (scalar_value - 1.0).abs() > f32::EPSILON {
                let kernel = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, intermediate_data_type)
                    .expect("Failed to create TensorAddScale kernel for layer_scalar");
                let zero_bias_arr =
                    context.create_array_zeros(&[model_dim], intermediate_data_type, "layer_scalar_zero_bias");
                let zero_bias_rc = zero_bias_arr.buffer();
                drop(zero_bias_arr);
                let zero_bias = Rc::try_unwrap(zero_bias_rc).expect("unique owner").into_inner();
                (scalar_value, Some(kernel), Some(zero_bias))
            } else {
                (scalar_value, None, None)
            }
        } else {
            (1.0, None, None)
        };

        Self {
            layer_index,
            #[cfg(feature = "tracing")]
            tensor_add,
            pre_attention_norm,
            mixer,
            post_attention_norm,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
            mlp_residual_add,
            ple,
            layer_scalar,
            layer_scalar_kernel,
            layer_scalar_zero_bias,
            model_dim,
            attention_num_heads: attn_num_heads,
            attention_num_groups: attn_num_groups,
            attention_head_dim: attn_head_dim,
            is_kv_shared_layer,
        }
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        #[cfg(feature = "tracing")]
        let layer_traces = state.traces().borrow().layer_results.get(self.layer_index).cloned();

        self.pre_attention_norm.encode(state, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(encoder, ArrayId::Shortcut, layer_traces.borrow().inputs.clone());
            state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().pre_attention_norm.clone());
        }

        // Buffers are allocated with max dims; each layer views the subset it needs.
        if matches!(&self.mixer, MixerExecutables::Attention { .. }) {
            let suffix_length = state.active_row_count();
            let num_heads = self.attention_num_heads;
            let num_groups = self.attention_num_groups;
            let head_dim = self.attention_head_dim;

            state.common_aux.rotated_queries =
                state.common_aux.rotated_queries.view(&[num_heads, suffix_length, head_dim]);
            state.common_aux.rotated_keys = state.common_aux.rotated_keys.view(&[num_groups, suffix_length, head_dim]);
            state.common_aux.extracted_values =
                state.common_aux.extracted_values.view(&[num_groups, suffix_length, head_dim]);
            state.common_aux.attention_output =
                state.common_aux.attention_output.view(&[suffix_length, num_heads * head_dim]);

            const TOTAL_BLOCKS_COUNT: usize = 32;
            state.common_aux.attention_partials =
                state.common_aux.attention_partials.view(&[num_heads * suffix_length * TOTAL_BLOCKS_COUNT * head_dim]);
            state.common_aux.attention_sums =
                state.common_aux.attention_sums.view(&[num_heads * suffix_length * TOTAL_BLOCKS_COUNT]);
            state.common_aux.attention_maxs =
                state.common_aux.attention_maxs.view(&[num_heads * suffix_length * TOTAL_BLOCKS_COUNT]);

            if let Some(ref gate) = state.common_aux.gate {
                state.common_aux.gate = Some(gate.view(&[suffix_length, num_heads * head_dim]));
            }
        }

        match &self.mixer {
            MixerExecutables::Attention {
                qkv_projection,
                gate_projection,
                qk_norm,
                rope,
                attention,
                out_projection,
            } => {
                qkv_projection.encode(state, encoder)?;
                if let Some(gate_proj) = gate_projection {
                    gate_proj.encode(state, encoder)?;
                }
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(encoder, ArrayId::QKV, layer_traces.borrow().qkv_projection.clone());
                }
                if let Some(norm) = qk_norm {
                    norm.encode(state, encoder)?;
                }
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(encoder, ArrayId::QKV, layer_traces.borrow().qk_norm.clone());
                }
                rope.encode(state, encoder)?;
                attention.encode(state, parameters, encoder, self.is_kv_shared_layer)?;
                out_projection.encode(state, encoder)?;
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().attention.clone());
                }
            },
            MixerExecutables::StateSpace {
                mixer,
            } => {
                mixer.encode(state, encoder)?;
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().attention.clone());
                }
            },
            MixerExecutables::ShortConv {
                mixer,
            } => {
                mixer.encode(state, encoder)?;
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().attention.clone());
                }
            },
            MixerExecutables::DeltaNet {
                mixer,
            } => {
                mixer.encode(state, encoder)?;
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().attention.clone());
                }
            },
        }

        if let Some(post_attention_norm) = &self.post_attention_norm {
            post_attention_norm.encode(state, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().post_attention_norm.clone());
            }
        }
        self.pre_mlp_norm.encode(state, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(encoder, ArrayId::Shortcut, layer_traces.borrow().mlp_inputs.clone());
            state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().pre_mlp_norm.clone());
        }

        self.mlp.encode(state, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().mlp.clone());
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            post_mlp_norm.encode(state, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().post_mlp_norm.clone());
            }
        }
        if let Some(ref residual_add) = self.mlp_residual_add {
            residual_add.encode(state, encoder)?;
        }

        if let Some(ref ple) = self.ple {
            ple.gate.encode(state, encoder)?;
            ple.activation.encode(state, encoder)?;

            // Element-wise multiply PleGate * PlePerLayerInputs[layer_i] (strided)
            {
                let ple_gate_arr = state.array(ArrayId::PleGate);
                let ple_inputs = state.array(ArrayId::PlePerLayerInputs);
                let rows = ple_gate_arr.shape()[0] as u32; // suffix_length

                // PleGate is both input_a and output (in-place mul).
                // This is safe because ElementWiseMulStrided reads input_a[row * ple_dim + col]
                // and writes output[row * ple_dim + col] — same index, element-wise.
                let ple_gate_buffer_rc = ple_gate_arr.buffer();
                let mut ple_gate_buffer = ple_gate_buffer_rc.borrow_mut();
                let ple_gate_input: &B::Buffer = unsafe { &*(&*ple_gate_buffer as *const B::Buffer) };

                ple.mul_strided_kernel.encode(
                    ple_gate_input,
                    &*ple_inputs.buffer().borrow(),
                    &mut *ple_gate_buffer,
                    ple.dim as u32,
                    ple.total_dim as u32,
                    (self.layer_index * ple.dim) as u32,
                    rows,
                    encoder,
                );
            }

            ple.projection.encode(state, encoder)?;
            ple.post_norm.encode(state, encoder)?;
            ple.add_swap.encode(state, encoder)?;
        }

        if let (Some(kernel), Some(zero_bias)) = (&self.layer_scalar_kernel, &self.layer_scalar_zero_bias) {
            let main = state.array(ArrayId::Main);
            let length = main.num_elements();

            let main_buffer_rc = main.buffer();
            let mut main_buffer = main_buffer_rc.borrow_mut();
            // TensorAddScale is element-wise, so in-place read/write aliasing is valid.
            let main_input: &B::Buffer = unsafe { &*(&*main_buffer as *const B::Buffer) };

            let num_cols = self.model_dim as u32;
            kernel.encode(
                (main_input, main.offset()),
                zero_bias,
                (&mut *main_buffer, main.offset()),
                num_cols,
                length as u32,
                self.layer_scalar,
                encoder,
            );
        }

        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            let size = state.array(ArrayId::Main).shape().into_iter().copied().product::<usize>() as u32;
            let output = layer_traces.borrow().outputs.buffer();

            if self.mlp_residual_add.is_some() {
                state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().outputs.clone());
            } else {
                let input_a = state.array(ArrayId::Main).buffer();
                let input_b = state.array(ArrayId::Shortcut).buffer();
                self.tensor_add.encode(
                    Some(input_a.borrow().deref()),
                    input_b.borrow().deref(),
                    output.borrow_mut().deref_mut(),
                    size,
                    size,
                    encoder,
                );
            }
        }

        Ok(())
    }
}
