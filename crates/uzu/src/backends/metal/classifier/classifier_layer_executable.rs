use std::rc::Rc;

use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::rc::autoreleasepool;

#[cfg(feature = "tracing")]
use super::ClassificationForwardPassState;
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
            AttentionKernelEncodable, KernelDataType, NormalizationEncodable,
            QKNormKernelEncodable, TensorAddSwap, TensorCopy,
        },
    },
    config::TransformerLayerConfig,
    parameters::ParameterTree,
};

pub struct ClassifierLayerExecutable {
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    layer_index: usize,
    copy_main_to_shortcut_mixer: Box<dyn EncodableWithState>,
    pre_attention_norm: Option<Box<dyn EncodableWithState>>,
    qkv_projection: Box<dyn EncodableWithState>,
    qk_norm: Option<Box<dyn EncodableWithState>>,
    rope: Rc<Box<dyn EncodableWithState>>,
    attention: Box<dyn EncodableWithState>,
    out_projection: Box<dyn EncodableWithState>,
    post_attention_norm: Option<Box<dyn EncodableWithState>>,
    mixer_residual_add: Box<dyn EncodableWithState>,
    copy_main_to_shortcut_mlp: Box<dyn EncodableWithState>,
    pre_mlp_norm: Box<dyn EncodableWithState>,
    mlp: Box<dyn EncodableWithState>,
    post_mlp_norm: Option<Box<dyn EncodableWithState>>,
    mlp_residual_add: Box<dyn EncodableWithState>,
}

impl ClassifierLayerExecutable {
    pub fn new(
        mtl_context: &MTLContext,
        layer_config: &TransformerLayerConfig,
        compilation_config: Rc<CompilationConfig>,
        layer_index: usize,
        model_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        attention_scale: Option<f32>,
        layer_loader: &ParameterTree<Rc<MTLContext>>,
        rope: Rc<Box<dyn EncodableWithState>>,
    ) -> Self {
        autoreleasepool(|_| {
            let intermediate_data_type: DataType = layer_config
                .attention_config
                .qkv_projection_config
                .activation_precision()
                .into();
            let kernel_data_type: KernelDataType =
                intermediate_data_type.into();

            let copy_main_to_shortcut_mixer: Box<dyn EncodableWithState> =
                Box::new(
                    TensorCopy::new(
                        mtl_context,
                        kernel_data_type,
                        vec![ArrayId::Main, ArrayId::Shortcut]
                            .into_boxed_slice(),
                    )
                    .unwrap(),
                );

            let pre_attention_norm: Option<Box<dyn EncodableWithState>> =
                if let Some(norm_config) =
                    &layer_config.pre_attention_norm_config
                {
                    if layer_loader.subtree("pre_mixer_norm").is_ok() {
                        Some(Box::new(
                            NormalizationEncodable::new(
                                mtl_context,
                                intermediate_data_type,
                                norm_config.clone(),
                                ArrayId::Main,
                                ArrayId::Main,
                                &layer_loader
                                    .subtree("pre_mixer_norm")
                                    .unwrap(),
                            )
                            .expect(
                                "Failed to create pre-attention norm kernel",
                            ),
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };

            let qkv_projection = transformer_layer::linear_block(
                &layer_config.attention_config.qkv_projection_config,
                layer_config.attention_config.has_qkv_biases,
                model_dim,
                [
                    num_heads * head_dim,
                    num_groups * head_dim,
                    num_groups * head_dim,
                ],
                mtl_context,
                &layer_loader.subtree("mixer.qkv_projection").unwrap(),
                ArrayId::Main,
                ArrayId::QKV,
                &compilation_config.descriptor_mlp,
            );

            let qk_norm: Option<Box<dyn EncodableWithState>> = if layer_config
                .attention_config
                .query_norm_config
                .is_some()
                || layer_config.attention_config.key_norm_config.is_some()
            {
                match QKNormKernelEncodable::new(
                    mtl_context,
                    intermediate_data_type,
                    layer_config.attention_config.query_norm_config.clone(),
                    layer_config.attention_config.key_norm_config.clone(),
                    ArrayId::QKV,
                    &layer_loader.subtree("mixer").unwrap(),
                    num_heads,
                    num_groups,
                    head_dim,
                ) {
                    Ok(norm) => Some(Box::new(norm)),
                    Err(e) => panic!(
                        "Failed to create QK norm kernel for layer {}: {:?}",
                        layer_index, e
                    ),
                }
            } else {
                None
            };

            let out_projection = transformer_layer::linear_block(
                &layer_config.attention_config.out_projection_config,
                layer_config.attention_config.has_out_biases,
                num_heads * head_dim,
                [model_dim],
                mtl_context,
                &layer_loader.subtree("mixer.out_projection").unwrap(),
                ArrayId::AttentionOutput,
                ArrayId::Main,
                &compilation_config.descriptor_mlp,
            );

            let post_attention_norm: Option<Box<dyn EncodableWithState>> =
                if let Some(norm_config) =
                    &layer_config.post_attention_norm_config
                {
                    Some(Box::new(
                        NormalizationEncodable::new(
                            mtl_context,
                            intermediate_data_type,
                            norm_config.clone(),
                            ArrayId::Main,
                            ArrayId::Main,
                            &layer_loader.subtree("post_mixer_norm").unwrap(),
                        )
                        .expect("Failed to create post-attention norm kernel"),
                    ))
                } else {
                    None
                };

            let mixer_residual_add: Box<dyn EncodableWithState> = Box::new(
                TensorAddSwap::new(
                    mtl_context,
                    kernel_data_type,
                    vec![ArrayId::Shortcut, ArrayId::Main].into_boxed_slice(),
                )
                .unwrap(),
            );

            let copy_main_to_shortcut_mlp: Box<dyn EncodableWithState> =
                Box::new(
                    TensorCopy::new(
                        mtl_context,
                        kernel_data_type,
                        vec![ArrayId::Main, ArrayId::Shortcut]
                            .into_boxed_slice(),
                    )
                    .unwrap(),
                );

            let pre_mlp_norm: Box<dyn EncodableWithState> = Box::new(
                NormalizationEncodable::new(
                    mtl_context,
                    intermediate_data_type,
                    layer_config.pre_mlp_norm_config.clone(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &layer_loader.subtree("pre_mlp_norm").unwrap(),
                )
                .expect("Failed to create pre-MLP norm kernel"),
            );

            let mlp = transformer_layer::mlp_block(
                &layer_config.mlp_config,
                model_dim,
                hidden_dim,
                mtl_context,
                &layer_loader.subtree("mlp").unwrap(),
                &compilation_config.descriptor_mlp,
            );

            let post_mlp_norm: Option<Box<dyn EncodableWithState>> =
                if let Some(norm_config) = &layer_config.post_mlp_norm_config {
                    Some(Box::new(
                        NormalizationEncodable::new(
                            mtl_context,
                            intermediate_data_type,
                            norm_config.clone(),
                            ArrayId::Main,
                            ArrayId::Main,
                            &layer_loader.subtree("post_mlp_norm").unwrap(),
                        )
                        .expect("Failed to create post-MLP norm kernel"),
                    ))
                } else {
                    None
                };

            let attention: Box<dyn EncodableWithState> = Box::new(
                AttentionKernelEncodable::new(
                    mtl_context,
                    kernel_data_type,
                    layer_index,
                    attention_scale,
                    layer_config.attention_config.has_sinks,
                    false,
                    layer_config.attention_config.sliding_window_size,
                )
                .expect("Failed to create attention kernel"),
            );

            let mlp_residual_add: Box<dyn EncodableWithState> = Box::new(
                TensorAddSwap::new(
                    mtl_context,
                    kernel_data_type,
                    vec![ArrayId::Shortcut, ArrayId::Main].into_boxed_slice(),
                )
                .unwrap(),
            );

            Self {
                layer_index,
                copy_main_to_shortcut_mixer,
                pre_attention_norm,
                qkv_projection,
                qk_norm,
                rope,
                attention,
                out_projection,
                post_attention_norm,
                mixer_residual_add,
                copy_main_to_shortcut_mlp,
                pre_mlp_norm,
                mlp,
                post_mlp_norm,
                mlp_residual_add,
            }
        })
    }
}

impl EncodableWithState for ClassifierLayerExecutable {
    fn encode(
        &self,
        state: &mut dyn ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        #[cfg(feature = "tracing")]
        let layer_traces = state
            .as_any()
            .downcast_ref::<ClassificationForwardPassState>()
            .and_then(|classifier_state| {
                classifier_state
                    .classifier_traces()
                    .borrow()
                    .layer_results
                    .get(self.layer_index)
                    .cloned()
            });

        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().inputs.clone(),
            );
        }

        self.copy_main_to_shortcut_mixer.encode(
            state,
            command_buffer,
            parameters,
        );

        if let Some(ref pre_attn_norm) = self.pre_attention_norm {
            pre_attn_norm.encode(state, command_buffer, parameters);
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(
                    command_buffer,
                    ArrayId::Main,
                    layer_traces.borrow().pre_attention_norm.clone(),
                );
            }
        } else {
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(
                    command_buffer,
                    ArrayId::Main,
                    layer_traces.borrow().pre_attention_norm.clone(),
                );
            }
        }

        self.qkv_projection.encode(state, command_buffer, parameters);
        if let Some(ref qk_norm) = self.qk_norm {
            qk_norm.encode(state, command_buffer, parameters);
        }
        self.rope.encode(state, command_buffer, parameters);
        self.attention.encode(state, command_buffer, parameters);
        self.out_projection.encode(state, command_buffer, parameters);
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().attention.clone(),
            );
        }

        if let Some(ref post_attn_norm) = self.post_attention_norm {
            post_attn_norm.encode(state, command_buffer, parameters);
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(
                    command_buffer,
                    ArrayId::Main,
                    layer_traces.borrow().post_attention_norm.clone(),
                );
            }
        }

        self.mixer_residual_add.encode(state, command_buffer, parameters);
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().mlp_inputs.clone(),
            );
        }

        self.copy_main_to_shortcut_mlp.encode(
            state,
            command_buffer,
            parameters,
        );

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

        if let Some(ref post_mlp_norm) = self.post_mlp_norm {
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

        self.mlp_residual_add.encode(state, command_buffer, parameters);
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(
                command_buffer,
                ArrayId::Main,
                layer_traces.borrow().outputs.clone(),
            );
        }
    }
}
