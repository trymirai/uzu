//! Layer executables - a single decoder layer with mixer, norms, and MLP.

use std::rc::Rc;

use objc2::rc::autoreleasepool;

use super::MixerExecutables;
use crate::{
    DataType, DecoderLayerConfig,
    backends::common::{Backend, CommandBuffer},
    config::{DecoderLayerType, MixerConfig},
    encodable_block::{
        Attention, EncodingParameters, Linear, MambaMixer, Mlp, QKNorm, RMSNorm, Rope, ShortConvMixer, TensorAddSwap,
        TensorCopy,
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

/// A single decoder layer with all its components.
pub struct LayerExecutables<B: Backend> {
    pub layer_index: usize,
    pub copy_main_to_shortcut: TensorCopy<B>,
    pub pre_attention_norm: RMSNorm<B>,
    pub(crate) mixer: MixerExecutables<B>,
    pub post_attention_norm: Option<RMSNorm<B>>,
    pub main_shortcut_add_swap: TensorAddSwap<B>,
    pub pre_mlp_norm: RMSNorm<B>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<RMSNorm<B>>,
}

impl<B: Backend> LayerExecutables<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: Rc<B::Context>,
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
        autoreleasepool(|_| {
            let ctx = context.as_ref(); // Reference for functions expecting &B::Context
            let intermediate_data_type: DataType = match &layer_config.mixer_config {
                MixerConfig::Attention(attention) => attention.qkv_projection_config.activation_precision().into(),
                MixerConfig::Mamba(mamba) => mamba.in_projection_config.activation_precision().into(),
                MixerConfig::ShortConv(short_conv) => short_conv.in_projection_config.activation_precision().into(),
                MixerConfig::DeltaNet(delta_net) => delta_net.in_proj_config.activation_precision().into(),
            };
            let copy_main_to_shortcut = TensorCopy::<B>::new(
                ctx,
                intermediate_data_type,
                vec![ArrayId::Main, ArrayId::Shortcut].into_boxed_slice(),
            )
            .unwrap();

            let pre_attention_norm = RMSNorm::new(
                ctx,
                intermediate_data_type,
                layer_config.pre_attention_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_layer_loader.subtree("pre_mixer_norm").unwrap(),
            )
            .expect("Failed to create RMS norm kernel");

            let mixer = match &layer_config.mixer_config {
                MixerConfig::Attention(attention_config) => {
                    let rope_block = rope.expect("RoPE encoder missing for attention layer");

                    if attention_config.has_gate || attention_config.partial_rope_dim.is_some() {
                        todo!("Gated attention and partial RoPE kernels not yet implemented");
                    }

                    let layer_num_heads = attention_config.num_heads.unwrap_or(num_heads);
                    let layer_num_groups = attention_config.num_groups.unwrap_or(num_groups);
                    let layer_head_dim = attention_config.head_dim.unwrap_or(head_dim);

                    let qkv_projection = <dyn Linear<B>>::new(
                        &attention_config.qkv_projection_config,
                        attention_config.has_qkv_biases,
                        model_dim,
                        [
                            layer_num_heads * layer_head_dim,
                            layer_num_groups * layer_head_dim,
                            layer_num_groups * layer_head_dim,
                        ],
                        ctx,
                        &decoder_layer_loader.subtree("mixer.qkv_projection").unwrap(),
                        ArrayId::Main,
                        ArrayId::QKV,
                    )
                    .expect("Failed to create qkv projection");

                    let qk_norm =
                        if attention_config.query_norm_config.is_some() || attention_config.key_norm_config.is_some() {
                            match QKNorm::new(
                                ctx,
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
                        layer_num_heads * layer_head_dim,
                        [model_dim],
                        ctx,
                        &decoder_layer_loader.subtree("mixer.out_projection").unwrap(),
                        ArrayId::AttentionOutput,
                        ArrayId::Main,
                    )
                    .expect("Failed to create out projection");

                    let attention = Attention::new(
                        ctx,
                        intermediate_data_type,
                        layer_index,
                        attention_scale,
                        attention_config.has_sinks,
                        attention_config.is_causal.unwrap_or(true),
                        attention_config.sliding_window_size,
                    )
                    .expect("Failed to create AttentionWrapper kernel");

                    MixerExecutables::Attention {
                        qkv_projection,
                        qk_norm,
                        rope: rope_block,
                        attention,
                        out_projection,
                    }
                },
                MixerConfig::Mamba(mamba_config) => {
                    let mixer = MambaMixer::new(
                        ctx,
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
                        ctx,
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
                MixerConfig::DeltaNet(_) => {
                    todo!("DeltaNet mixer kernel not yet implemented");
                },
            };

            let post_attention_norm = if let Some(norm_config) = &layer_config.post_attention_norm_config {
                Some(
                    RMSNorm::new(
                        ctx,
                        intermediate_data_type,
                        norm_config.clone(),
                        ArrayId::Main,
                        ArrayId::Main,
                        &decoder_layer_loader.subtree("post_mixer_norm").unwrap(),
                    )
                    .expect("Failed to create RMS norm kernel"),
                )
            } else {
                None
            };

            let main_shortcut_add_swap = TensorAddSwap::<B>::new(
                ctx,
                intermediate_data_type,
                vec![ArrayId::Shortcut, ArrayId::Main].into_boxed_slice(),
            )
            .unwrap();

            let pre_mlp_norm = RMSNorm::new(
                ctx,
                intermediate_data_type,
                layer_config.pre_mlp_norm_config.clone(),
                ArrayId::Main,
                ArrayId::Main,
                &decoder_layer_loader.subtree("pre_mlp_norm").unwrap(),
            )
            .expect("Failed to create RMS norm kernel");

            let mlp = <dyn Mlp<B>>::new(
                &layer_config.mlp_config,
                model_dim,
                hidden_dim,
                context.as_ref(),
                &decoder_layer_loader.subtree("mlp").unwrap(),
            )
            .expect("Failed to create mlp block");

            let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
                Some(
                    RMSNorm::new(
                        ctx,
                        intermediate_data_type,
                        norm_config.clone(),
                        ArrayId::Main,
                        ArrayId::Main,
                        &decoder_layer_loader.subtree("post_mlp_norm").unwrap(),
                    )
                    .expect("Failed to create RMS norm kernel"),
                )
            } else {
                None
            };

            Self {
                layer_index,
                copy_main_to_shortcut,
                pre_attention_norm,
                mixer,
                post_attention_norm,
                main_shortcut_add_swap,
                pre_mlp_norm,
                mlp,
                post_mlp_norm,
            }
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        #[cfg(feature = "tracing")]
        let layer_traces = state.traces().borrow().layer_results.get(self.layer_index).cloned();

        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().inputs.clone());
        }

        self.copy_main_to_shortcut.encode(state, command_buffer)?;
        // shortcut = input

        self.pre_attention_norm.encode(state, command_buffer)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().pre_attention_norm.clone());
        }

        match &self.mixer {
            MixerExecutables::Attention {
                qkv_projection,
                qk_norm,
                rope,
                attention,
                out_projection,
            } => {
                qkv_projection.encode(state, command_buffer)?;
                if let Some(norm) = qk_norm {
                    norm.encode(state, command_buffer)?;
                }
                rope.encode(state, command_buffer)?;
                attention.encode(state, parameters, command_buffer)?;
                out_projection.encode(state, command_buffer)?;
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().attention.clone());
                }
            },
            MixerExecutables::StateSpace {
                mixer,
            } => {
                mixer.encode(state, command_buffer)?;
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().attention.clone());
                }
            },
            MixerExecutables::ShortConv {
                mixer,
            } => {
                mixer.encode(state, command_buffer)?;
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().attention.clone());
                }
            },
        }

        if let Some(post_attention_norm) = &self.post_attention_norm {
            post_attention_norm.encode(state, command_buffer)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(
                    command_buffer,
                    ArrayId::Main,
                    layer_traces.borrow().post_attention_norm.clone(),
                );
            }
        }
        //main = attention_result

        self.main_shortcut_add_swap.encode(state, command_buffer)?;
        // shortcut = input + attention_result
        // main = input + attention_result
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().mlp_inputs.clone());
        }

        self.pre_mlp_norm.encode(state, command_buffer)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().pre_mlp_norm.clone());
        }

        self.mlp.encode(state, command_buffer)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().mlp.clone());
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            post_mlp_norm.encode(state, command_buffer)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().post_mlp_norm.clone());
            }
        }
        // main = mlp_result

        self.main_shortcut_add_swap.encode(state, command_buffer)?;
        // shortcut = input + attention_result + mlp_result
        // main = input + attention_result + mlp_result
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(command_buffer, ArrayId::Main, layer_traces.borrow().outputs.clone());
        }

        Ok(())
    }
}
