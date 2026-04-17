//! Layer executables - a single decoder layer with mixer, norms, and MLP.

#[cfg(feature = "tracing")]
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use super::MixerExecutables;
#[cfg(feature = "tracing")]
use crate::backends::common::{Kernels, kernel::TensorAddBiasKernel};
use crate::{
    DataType,
    backends::common::{Backend, Encoder},
    config::{DecoderLayerConfig, DecoderLayerType, MixerConfig},
    encodable_block::{
        Attention, DeltaNetMixer, EncodingParameters, Linear, MambaMixer, Mlp, QKNorm, RMSNorm, Rope, ShortConvMixer,
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

/// A single decoder layer with all its components.
pub struct LayerExecutables<B: Backend> {
    #[cfg(feature = "tracing")]
    pub layer_index: usize,
    #[cfg(feature = "tracing")]
    pub tensor_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    pub pre_attention_norm: RMSNorm<B>,
    pub(crate) mixer: MixerExecutables<B>,
    pub post_attention_norm: Option<RMSNorm<B>>,
    pub pre_mlp_norm: RMSNorm<B>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<RMSNorm<B>>,
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

        // (mixer, input_hadamard_factors, pre_attention_adapter_down_prime)
        let (mixer, mixer_hadamard_factors, mixer_adapter_down_prime) = match &layer_config.mixer_config {
            MixerConfig::Attention(attention_config) => {
                let rope_block = rope.expect("RoPE encoder missing for attention layer");
                let use_rope = attention_config.use_rope;

                let layer_num_heads = attention_config.num_heads.unwrap_or(num_heads);
                let layer_num_groups = attention_config.num_groups.unwrap_or(num_groups);
                let layer_head_dim = attention_config.head_dim.unwrap_or(head_dim);

                let q_dim = layer_num_heads * layer_head_dim;
                let kv_dim = layer_num_groups * layer_head_dim;

                let (qkv_projection, input_hadamard_factors, qkv_adapter_down_prime) =
                    <dyn Linear<B>>::new_extracting_input_hadamard(
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

                let qk_norm =
                    if attention_config.query_norm_config.is_some() || attention_config.key_norm_config.is_some() {
                        match QKNorm::new(
                            context,
                            intermediate_data_type,
                            attention_config.query_norm_config.clone(),
                            attention_config.key_norm_config.clone(),
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

                (
                    MixerExecutables::Attention {
                        qkv_projection,
                        gate_projection,
                        qk_norm,
                        rope: rope_block,
                        use_rope,
                        attention,
                        out_projection,
                    },
                    input_hadamard_factors,
                    qkv_adapter_down_prime,
                )
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
                (
                    MixerExecutables::StateSpace {
                        mixer,
                    },
                    None,
                    None,
                )
            },
            MixerConfig::ShortConv(short_conv_config) => {
                let (mixer, input_hadamard_factors) = ShortConvMixer::new(
                    context,
                    layer_type.clone(),
                    short_conv_config.clone(),
                    layer_index,
                    model_dim,
                    decoder_layer_loader,
                );
                (
                    MixerExecutables::ShortConv {
                        mixer,
                    },
                    input_hadamard_factors,
                    None,
                )
            },
            MixerConfig::DeltaNet(delta_net_config) => {
                let mixer =
                    DeltaNetMixer::new(context, delta_net_config.clone(), layer_index, model_dim, decoder_layer_loader)
                        .expect("Failed to create DeltaNet mixer");
                (
                    MixerExecutables::DeltaNet {
                        mixer,
                    },
                    None,
                    None,
                )
            },
        };

        let pre_attention_norm = if let Some(adapter_down_prime) = mixer_adapter_down_prime {
            RMSNorm::new_with_lora_fuse(
                context,
                intermediate_data_type,
                layer_config.pre_attention_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_layer_loader.subtree("pre_mixer_norm").unwrap(),
                mixer_hadamard_factors,
                Some(ArrayId::Shortcut),
                layer_index > 0,
                adapter_down_prime,
            )
            .expect("Failed to create RMS norm kernel with LoRA fuse")
        } else {
            RMSNorm::new(
                context,
                intermediate_data_type,
                layer_config.pre_attention_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_layer_loader.subtree("pre_mixer_norm").unwrap(),
                mixer_hadamard_factors,
                Some(ArrayId::Shortcut),
                layer_index > 0,
            )
            .expect("Failed to create RMS norm kernel")
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
                    None,
                    false,
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
        };

        let (mlp, mlp_input_hadamard_factors, mlp_adapter_down_prime) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            model_dim,
            hidden_dim,
            context,
            &decoder_layer_loader.subtree("mlp").unwrap(),
        )
        .expect("Failed to create mlp block");

        let pre_mlp_norm = if let Some(adapter_down_prime) = mlp_adapter_down_prime {
            RMSNorm::new_with_lora_fuse(
                context,
                intermediate_data_type,
                layer_config.pre_mlp_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_layer_loader.subtree("pre_mlp_norm").unwrap(),
                mlp_input_hadamard_factors,
                Some(ArrayId::Shortcut),
                true,
                adapter_down_prime,
            )
            .expect("Failed to create pre-MLP RMS norm kernel with LoRA fuse")
        } else {
            RMSNorm::new(
                context,
                intermediate_data_type,
                layer_config.pre_mlp_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_layer_loader.subtree("pre_mlp_norm").unwrap(),
                mlp_input_hadamard_factors,
                Some(ArrayId::Shortcut),
                true,
            )
            .expect("Failed to create RMS norm kernel")
        };

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
                    None,
                    false,
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
        };

        Self {
            #[cfg(feature = "tracing")]
            layer_index,
            #[cfg(feature = "tracing")]
            tensor_add,
            pre_attention_norm,
            mixer,
            post_attention_norm,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
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

        match &self.mixer {
            MixerExecutables::Attention {
                qkv_projection,
                gate_projection,
                qk_norm,
                rope,
                use_rope,
                attention,
                out_projection,
            } => {
                qkv_projection.encode(state, encoder)?;
                if let Some(gate_proj) = gate_projection {
                    gate_proj.encode(state, encoder)?;
                }
                if let Some(norm) = qk_norm {
                    norm.encode(state, encoder)?;
                }
                rope.encode(state, *use_rope, encoder)?;
                attention.encode(state, parameters, encoder)?;
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
        // main = attention_result

        self.pre_mlp_norm.encode(state, encoder)?;
        // shortcut = attention_result + shortcut; main = normalized(shortcut)
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
        // main = mlp_result
        // next layer's pre_attention_norm (or output_norm) fuses the residual add

        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            let size = state.array(ArrayId::Main).shape().into_iter().copied().product::<usize>() as u32;
            let input_a = state.array(ArrayId::Main).buffer();
            let input_b = state.array(ArrayId::Shortcut).buffer();
            let output = layer_traces.borrow().outputs.buffer();
            self.tensor_add.encode(
                Some(input_a.borrow().deref()),
                input_b.borrow().deref(),
                output.borrow_mut().deref_mut(),
                size,
                size,
                encoder,
            );
        }

        Ok(())
    }
}
