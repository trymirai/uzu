//! Layer executables - a single decoder layer with mixer, norms, and MLP.

use crate::backends::metal::Metal;
use std::rc::Rc;

#[cfg(not(feature = "tracing"))]
use crate::backends::metal::MTLCommandEncoder;
use crate::backends::metal::{
    MTLCommandBuffer, MTLComputeCommandEncoder, ProtocolObject, Retained,
};
use objc2::rc::autoreleasepool;

use super::{
    super::{
        Attention, EncodableBlock, EncodingParameters, MambaMixer, QKNorm,
        RMSNorm, ShortConvMixer, TensorAddSwap, TensorCopy, transformer_layer,
    },
    MixerExecutables,
};
use crate::{
    DataType, DecoderLayerConfig,
    backends::metal::{
        MTLContext,
        compilation_parameters::CompilationConfig,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::KernelDataType,
    },
    config::{DecoderLayerType, MixerConfig},
    parameters::ParameterTree,
};

/// A single decoder layer with all its components.
pub struct LayerExecutables {
    pub layer_index: usize,
    pub copy_main_to_shortcut: Box<dyn EncodableBlock<Metal>>,
    pub pre_attention_norm: Box<dyn EncodableBlock<Metal>>,
    pub(crate) mixer: MixerExecutables,
    pub post_attention_norm: Option<Box<dyn EncodableBlock<Metal>>>,
    pub main_shortcut_add_swap: Box<dyn EncodableBlock<Metal>>,
    pub pre_mlp_norm: Box<dyn EncodableBlock<Metal>>,
    pub mlp: Box<dyn EncodableBlock<Metal>>,
    pub post_mlp_norm: Option<Box<dyn EncodableBlock<Metal>>>,
}

impl LayerExecutables {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mtl_context: Rc<MTLContext>,
        layer_config: &DecoderLayerConfig,
        layer_type: &DecoderLayerType,
        compilation_config: Rc<CompilationConfig>,
        layer_index: usize,
        model_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        attention_scale: Option<f32>,
        decoder_layer_loader: &ParameterTree<MTLContext>,
        rope: Option<Rc<Box<dyn EncodableBlock<Metal>>>>,
    ) -> Self {
        autoreleasepool(|_| {
            let ctx = &*mtl_context; // Reference for functions expecting &MTLContext
            let intermediate_data_type: DataType =
                match &layer_config.mixer_config {
                    MixerConfig::Attention(attention) => attention
                        .qkv_projection_config
                        .activation_precision()
                        .into(),
                    MixerConfig::Mamba(mamba) => {
                        mamba.in_projection_config.activation_precision().into()
                    },
                    MixerConfig::ShortConv(short_conv) => short_conv
                        .in_projection_config
                        .activation_precision()
                        .into(),
                };
            let kernel_data_type: KernelDataType =
                intermediate_data_type.into();

            let copy_main_to_shortcut: Box<dyn EncodableBlock<Metal>> =
                Box::new(
                    TensorCopy::new(
                        ctx,
                        kernel_data_type,
                        vec![ArrayId::Main, ArrayId::Shortcut]
                            .into_boxed_slice(),
                    )
                    .unwrap(),
                );

            let pre_attention_norm: Box<dyn EncodableBlock<Metal>> = Box::new(
                RMSNorm::new(
                    ctx,
                    intermediate_data_type,
                    layer_config.pre_attention_norm_config.clone(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &decoder_layer_loader.subtree("pre_mixer_norm").unwrap(),
                )
                .expect("Failed to create RMS norm kernel"),
            );

            let mixer = match &layer_config.mixer_config {
                MixerConfig::Attention(attention_config) => {
                    let rope_block = rope
                        .clone()
                        .expect("RoPE encoder missing for attention layer");

                    let qkv_projection = transformer_layer::linear_block(
                        &attention_config.qkv_projection_config,
                        attention_config.has_qkv_biases,
                        model_dim,
                        [
                            num_heads * head_dim,
                            num_groups * head_dim,
                            num_groups * head_dim,
                        ],
                        ctx,
                        &decoder_layer_loader
                            .subtree("mixer.qkv_projection")
                            .unwrap(),
                        ArrayId::Main,
                        ArrayId::QKV,
                    )
                    .expect("Failed to create qkv projection");

                    let qk_norm: Option<Box<dyn EncodableBlock<Metal>>> =
                        if attention_config.query_norm_config.is_some()
                            || attention_config.key_norm_config.is_some()
                        {
                            match QKNorm::new(
                                ctx,
                                intermediate_data_type,
                                attention_config.query_norm_config.clone(),
                                attention_config.key_norm_config.clone(),
                                ArrayId::QKV,
                                &decoder_layer_loader.subtree("mixer").unwrap(),
                                num_heads,
                                num_groups,
                                head_dim,
                            ) {
                                Ok(qk_norm) => Some(Box::new(qk_norm)
                                    as Box<dyn EncodableBlock<Metal>>),
                                Err(e) => panic!(
                                    "Failed to create QK norm kernel for layer {}: {:?}",
                                    layer_index, e
                                ),
                            }
                        } else {
                            None
                        };

                    let out_projection = transformer_layer::linear_block(
                        &attention_config.out_projection_config,
                        attention_config.has_out_biases,
                        num_heads * head_dim,
                        [model_dim],
                        ctx,
                        &decoder_layer_loader
                            .subtree("mixer.out_projection")
                            .unwrap(),
                        ArrayId::AttentionOutput,
                        ArrayId::Main,
                    )
                    .expect("Failed to create out projection");

                    let attention = Box::new(
                        Attention::new(
                            ctx,
                            kernel_data_type,
                            layer_index,
                            attention_scale,
                            attention_config.has_sinks,
                            attention_config.is_causal.unwrap_or(true),
                            attention_config.sliding_window_size,
                        )
                        .expect("Failed to create AttentionWrapper with Metal kernel"),
                    );

                    MixerExecutables::Attention {
                        qkv_projection,
                        qk_norm,
                        rope: rope_block,
                        attention,
                        out_projection,
                    }
                },
                MixerConfig::Mamba(mamba_config) => {
                    let mixer: Box<dyn EncodableBlock<Metal>> =
                        Box::new(MambaMixer::new(
                            ctx,
                            layer_type.clone(),
                            mamba_config.clone(),
                            compilation_config.clone(),
                            layer_index,
                            model_dim,
                            num_heads,
                            head_dim,
                            num_groups,
                            decoder_layer_loader,
                        ));
                    MixerExecutables::StateSpace {
                        mixer,
                    }
                },
                MixerConfig::ShortConv(short_conv_config) => {
                    let mixer: Box<dyn EncodableBlock<Metal>> =
                        Box::new(ShortConvMixer::new(
                            ctx,
                            layer_type.clone(),
                            short_conv_config.clone(),
                            compilation_config.clone(),
                            layer_index,
                            model_dim,
                            decoder_layer_loader,
                        ));
                    MixerExecutables::ShortConv {
                        mixer,
                    }
                },
            };

            let post_attention_norm: Option<Box<dyn EncodableBlock<Metal>>> =
                if let Some(norm_config) =
                    &layer_config.post_attention_norm_config
                {
                    Some(Box::new(
                        RMSNorm::new(
                            ctx,
                            intermediate_data_type,
                            norm_config.clone(),
                            ArrayId::Main,
                            ArrayId::Main,
                            &decoder_layer_loader
                                .subtree("post_mixer_norm")
                                .unwrap(),
                        )
                        .expect("Failed to create RMS norm kernel"),
                    ))
                } else {
                    None
                };

            let main_shortcut_add_swap: Box<dyn EncodableBlock<Metal>> =
                Box::new(
                    TensorAddSwap::new(
                        ctx,
                        kernel_data_type,
                        vec![ArrayId::Shortcut, ArrayId::Main]
                            .into_boxed_slice(),
                    )
                    .unwrap(),
                );

            let pre_mlp_norm: Box<dyn EncodableBlock<Metal>> = Box::new(
                RMSNorm::new(
                    ctx,
                    intermediate_data_type,
                    layer_config.pre_mlp_norm_config.clone(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &decoder_layer_loader.subtree("pre_mlp_norm").unwrap(),
                )
                .expect("Failed to create RMS norm kernel"),
            );

            let mlp = transformer_layer::mlp_block(
                &layer_config.mlp_config,
                model_dim,
                hidden_dim,
                &mtl_context,
                &decoder_layer_loader.subtree("mlp").unwrap(),
            )
            .expect("Failed to create mlp block");

            let post_mlp_norm: Option<Box<dyn EncodableBlock<Metal>>> =
                if let Some(norm_config) = &layer_config.post_mlp_norm_config {
                    Some(Box::new(
                        RMSNorm::new(
                            ctx,
                            intermediate_data_type,
                            norm_config.clone(),
                            ArrayId::Main,
                            ArrayId::Main,
                            &decoder_layer_loader
                                .subtree("post_mlp_norm")
                                .unwrap(),
                        )
                        .expect("Failed to create RMS norm kernel"),
                    ))
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
}

impl EncodableBlock<Metal> for LayerExecutables {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters,
    ) {
        // In non-tracing builds, if every sub-block supports shared encoding,
        // we can run the entire layer in a single compute encoder.
        #[cfg(not(feature = "tracing"))]
        {
            if self.supports_shared_encoder() {
                let encoder = command_buffer
                    .new_compute_command_encoder()
                    .expect("Failed to create compute command encoder");
                self.encode_with_shared_encoder(state, &encoder, parameters);
                encoder.end_encoding();
                return;
            }
        }

        #[cfg(feature = "tracing")]
        let layer_traces = state
            .traces()
            .borrow()
            .layer_results
            .get(self.layer_index)
            .cloned();

        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().inputs.clone(),
            );
        }

        self.copy_main_to_shortcut.encode(state, command_buffer, parameters);
        // shortcut = input

        self.pre_attention_norm.encode(state, command_buffer, parameters);
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().pre_attention_norm.clone(),
            );
        }

        match &self.mixer {
            MixerExecutables::Attention {
                qkv_projection,
                qk_norm,
                rope,
                attention,
                out_projection,
            } => {
                qkv_projection.encode(state, command_buffer, parameters);
                if let Some(norm) = qk_norm {
                    norm.encode(state, command_buffer, parameters);
                }
                rope.encode(state, command_buffer, parameters);
                attention.encode(state, command_buffer, parameters);
                out_projection.encode(state, command_buffer, parameters);
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(
                        command_buffer,
                        ArrayId::Main,
                        layer_traces.borrow().attention.clone(),
                    );
                }
            },
            MixerExecutables::StateSpace {
                mixer,
            } => {
                mixer.encode(state, command_buffer, parameters);
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(
                        command_buffer,
                        ArrayId::Main,
                        layer_traces.borrow().attention.clone(),
                    );
                }
            },
            MixerExecutables::ShortConv {
                mixer,
            } => {
                mixer.encode(state, command_buffer, parameters);
                #[cfg(feature = "tracing")]
                if let Some(ref layer_traces) = layer_traces {
                    state.encode_copy_array(
                        command_buffer,
                        ArrayId::Main,
                        layer_traces.borrow().attention.clone(),
                    );
                }
            },
        }

        if let Some(post_attention_norm) = &self.post_attention_norm {
            post_attention_norm.encode(state, command_buffer, parameters);
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

        self.main_shortcut_add_swap.encode(state, command_buffer, parameters);
        // shortcut = input + attention_result
        // main = input + attention_result
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().mlp_inputs.clone(),
            );
        }

        self.pre_mlp_norm.encode(state, command_buffer, parameters);
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().pre_mlp_norm.clone(),
            );
        }

        self.mlp.encode(state, command_buffer, parameters);
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().mlp.clone(),
            );
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            post_mlp_norm.encode(state, command_buffer, parameters);
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(
                    command_buffer,
                    ArrayId::Main,
                    layer_traces.borrow().post_mlp_norm.clone(),
                );
            }
        }
        // main = mlp_result

        self.main_shortcut_add_swap.encode(state, command_buffer, parameters);
        // shortcut = input + attention_result + mlp_result
        // main = input + attention_result + mlp_result
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().outputs.clone(),
            );
        }

        let _ = parameters;
    }

    fn supports_shared_encoder(&self) -> bool {
        #[cfg(feature = "tracing")]
        {
            false
        }

        #[cfg(not(feature = "tracing"))]
        {
            let mixer_ok = match &self.mixer {
                MixerExecutables::Attention {
                    qkv_projection,
                    qk_norm,
                    rope,
                    attention,
                    out_projection,
                } => {
                    qkv_projection.supports_shared_encoder()
                        && qk_norm
                            .as_ref()
                            .map(|b| b.supports_shared_encoder())
                            .unwrap_or(true)
                        && rope.supports_shared_encoder()
                        && attention.supports_shared_encoder()
                        && out_projection.supports_shared_encoder()
                },
                MixerExecutables::StateSpace {
                    mixer,
                } => mixer.supports_shared_encoder(),
                MixerExecutables::ShortConv {
                    mixer,
                } => mixer.supports_shared_encoder(),
            };

            self.copy_main_to_shortcut.supports_shared_encoder()
                && self.pre_attention_norm.supports_shared_encoder()
                && mixer_ok
                && self
                    .post_attention_norm
                    .as_ref()
                    .map(|b| b.supports_shared_encoder())
                    .unwrap_or(true)
                && self.main_shortcut_add_swap.supports_shared_encoder()
                && self.pre_mlp_norm.supports_shared_encoder()
                && self.mlp.supports_shared_encoder()
                && self
                    .post_mlp_norm
                    .as_ref()
                    .map(|b| b.supports_shared_encoder())
                    .unwrap_or(true)
        }
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        parameters: &EncodingParameters,
    ) {
        debug_assert!(
            self.supports_shared_encoder(),
            "encode_with_shared_encoder called on unsupported LayerExecutables"
        );

        self.copy_main_to_shortcut
            .encode_with_shared_encoder(state, encoder, parameters);
        self.pre_attention_norm
            .encode_with_shared_encoder(state, encoder, parameters);

        match &self.mixer {
            MixerExecutables::Attention {
                qkv_projection,
                qk_norm,
                rope,
                attention,
                out_projection,
            } => {
                qkv_projection
                    .encode_with_shared_encoder(state, encoder, parameters);
                if let Some(norm) = qk_norm {
                    norm.encode_with_shared_encoder(state, encoder, parameters);
                }
                rope.encode_with_shared_encoder(state, encoder, parameters);
                attention
                    .encode_with_shared_encoder(state, encoder, parameters);
                out_projection
                    .encode_with_shared_encoder(state, encoder, parameters);
            },
            MixerExecutables::StateSpace {
                mixer,
            } => {
                mixer.encode_with_shared_encoder(state, encoder, parameters);
            },
            MixerExecutables::ShortConv {
                mixer,
            } => {
                mixer.encode_with_shared_encoder(state, encoder, parameters);
            },
        }

        if let Some(post_attention_norm) = &self.post_attention_norm {
            post_attention_norm
                .encode_with_shared_encoder(state, encoder, parameters);
        }

        self.main_shortcut_add_swap
            .encode_with_shared_encoder(state, encoder, parameters);

        self.pre_mlp_norm
            .encode_with_shared_encoder(state, encoder, parameters);
        self.mlp.encode_with_shared_encoder(state, encoder, parameters);

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            post_mlp_norm
                .encode_with_shared_encoder(state, encoder, parameters);
        }

        self.main_shortcut_add_swap
            .encode_with_shared_encoder(state, encoder, parameters);
    }
}
