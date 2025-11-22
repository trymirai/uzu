use std::rc::Rc;

use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::rc::autoreleasepool;

use crate::{
    DataType,
    backends::metal::{
        MTLContext,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
            transformer_layer,
        },
        kernel::{
            AttentionKernelEncodable, KernelDataType, MambaMixerEncodable,
            QKNormKernelEncodable, RMSNormKernelEncodable, TensorAddSwap,
            TensorCopy,
        },
    },
    config::{
        DecoderLayerType,
        decoder_layer::{DecoderLayerConfig, MixerConfig},
    },
    parameters::ParameterTree,
};

pub struct LayerExecutables {
    pub layer_index: usize,
    pub copy_main_to_shortcut: Box<dyn EncodableWithState>,
    pub pre_attention_norm: Box<dyn EncodableWithState>,
    pub(crate) mixer: MixerExecutables,
    pub post_attention_norm: Option<Box<dyn EncodableWithState>>,
    pub main_shortcut_add_swap: Box<dyn EncodableWithState>,
    pub pre_mlp_norm: Box<dyn EncodableWithState>,
    pub mlp: Box<dyn EncodableWithState>,
    pub post_mlp_norm: Option<Box<dyn EncodableWithState>>,
}

pub(crate) enum MixerExecutables {
    Attention {
        qkv_projection: Box<dyn EncodableWithState>,
        qk_norm: Option<Box<dyn EncodableWithState>>,
        rope: Rc<Box<dyn EncodableWithState>>,
        attention: Box<dyn EncodableWithState>,
        out_projection: Box<dyn EncodableWithState>,
    },
    StateSpace {
        mixer: Box<dyn EncodableWithState>,
    },
}

impl LayerExecutables {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mtl_context: &MTLContext,
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
        decoder_layer_loader: &ParameterTree<Rc<MTLContext>>,
        rope: Option<Rc<Box<dyn EncodableWithState>>>,
    ) -> Self {
        autoreleasepool(|_| {
            let intermediate_data_type: DataType =
                match &layer_config.mixer_config {
                    MixerConfig::Attention(attention) => attention
                        .qkv_projection_config
                        .activation_precision()
                        .into(),
                    MixerConfig::Mamba(mamba) => {
                        mamba.in_projection_config.activation_precision().into()
                    },
                };
            let kernel_data_type: KernelDataType =
                intermediate_data_type.into();

            let copy_main_to_shortcut: Box<dyn EncodableWithState> = Box::new(
                TensorCopy::new(
                    mtl_context,
                    kernel_data_type,
                    vec![ArrayId::Main, ArrayId::Shortcut].into_boxed_slice(),
                )
                .unwrap(),
            );

            let pre_attention_norm: Box<dyn EncodableWithState> = Box::new(
                RMSNormKernelEncodable::new(
                    mtl_context,
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
                        mtl_context,
                        &decoder_layer_loader
                            .subtree("mixer.qkv_projection")
                            .unwrap(),
                        ArrayId::Main,
                        ArrayId::QKV,
                        &compilation_config.descriptor_mlp,
                    );

                    let qk_norm: Option<Box<dyn EncodableWithState>> =
                        if attention_config.query_norm_config.is_some()
                            || attention_config.key_norm_config.is_some()
                        {
                            match QKNormKernelEncodable::new(
                                mtl_context,
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
                                    as Box<dyn EncodableWithState>),
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
                        mtl_context,
                        &decoder_layer_loader
                            .subtree("mixer.out_projection")
                            .unwrap(),
                        ArrayId::AttentionOutput,
                        ArrayId::Main,
                        &compilation_config.descriptor_mlp,
                    );

                    let attention = Box::new(
                        AttentionKernelEncodable::new(
                            mtl_context,
                            kernel_data_type,
                            layer_index,
                            attention_scale,
                            attention_config.has_sinks,
                        )
                        .expect(
                            "Failed to create AttentionWrapper with Metal kernel",
                        ),
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
                    let mixer: Box<dyn EncodableWithState> =
                        Box::new(MambaMixerEncodable::new(
                            mtl_context,
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
            };

            let post_attention_norm: Option<Box<dyn EncodableWithState>> =
                if let Some(norm_config) =
                    &layer_config.post_attention_norm_config
                {
                    Some(Box::new(
                        RMSNormKernelEncodable::new(
                            mtl_context,
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

            let main_shortcut_add_swap: Box<dyn EncodableWithState> = Box::new(
                TensorAddSwap::new(
                    mtl_context,
                    kernel_data_type,
                    vec![ArrayId::Shortcut, ArrayId::Main].into_boxed_slice(),
                )
                .unwrap(),
            );

            let pre_mlp_norm: Box<dyn EncodableWithState> = Box::new(
                RMSNormKernelEncodable::new(
                    mtl_context,
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
                mtl_context,
                &decoder_layer_loader.subtree("mlp").unwrap(),
                &compilation_config.descriptor_mlp,
            );

            let post_mlp_norm: Option<Box<dyn EncodableWithState>> =
                if let Some(norm_config) = &layer_config.post_mlp_norm_config {
                    Some(Box::new(
                        RMSNormKernelEncodable::new(
                            mtl_context,
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

impl EncodableWithState for LayerExecutables {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let layer_traces = if let Some(traces) = state.traces.clone() {
            traces.borrow().layer_results.get(self.layer_index).cloned()
        } else {
            None
        };

        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(
                ArrayId::Main,
                layer_traces.borrow().inputs.clone(),
            );
        }

        self.copy_main_to_shortcut.encode(state, command_buffer, parameters);
        // shortcut = input

        self.pre_attention_norm.encode(state, command_buffer, parameters);
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(
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
                if let Some(layer_traces) = layer_traces.clone() {
                    state.copy_array(
                        ArrayId::Main,
                        layer_traces.borrow().attention.clone(),
                    );
                }
            },
            MixerExecutables::StateSpace {
                mixer,
            } => {
                mixer.encode(state, command_buffer, parameters);
                if let Some(layer_traces) = layer_traces.clone() {
                    state.copy_array(
                        ArrayId::Main,
                        layer_traces.borrow().attention.clone(),
                    );
                }
            },
        }

        if let Some(post_attention_norm) = &self.post_attention_norm {
            post_attention_norm.encode(state, command_buffer, parameters);
            if let Some(layer_traces) = layer_traces.clone() {
                state.copy_array(
                    ArrayId::Main,
                    layer_traces.borrow().post_attention_norm.clone(),
                );
            }
        }
        //main = attention_result

        self.main_shortcut_add_swap.encode(state, command_buffer, parameters);
        // shortcut = input + attention_result
        // main = input + attention_result
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(
                ArrayId::Main,
                layer_traces.borrow().mlp_inputs.clone(),
            );
        }

        self.pre_mlp_norm.encode(state, command_buffer, parameters);
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(
                ArrayId::Main,
                layer_traces.borrow().pre_mlp_norm.clone(),
            );
        }

        self.mlp.encode(state, command_buffer, parameters);
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(ArrayId::Main, layer_traces.borrow().mlp.clone());
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            post_mlp_norm.encode(state, command_buffer, parameters);
            if let Some(layer_traces) = layer_traces.clone() {
                state.copy_array(
                    ArrayId::Main,
                    layer_traces.borrow().post_mlp_norm.clone(),
                );
            }
        }
        // main = mlp_result

        self.main_shortcut_add_swap.encode(state, command_buffer, parameters);
        // shortcut = input + attention_result + mlp_result
        // main = input + attention_result + mlp_result
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(
                ArrayId::Main,
                layer_traces.borrow().outputs.clone(),
            );
        }
    }
}
