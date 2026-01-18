use std::rc::Rc;

use metal::{CommandBufferRef, ComputeCommandEncoderRef};
use objc2::rc::autoreleasepool;

use super::{
    Attention, EncodableBlock, EncodingParameters, ForwardPassState,
    Normalization, QKNorm, TensorAddSwap, TensorCopy, transformer_layer,
};
use crate::{
    DataType,
    backends::metal::{
        MTLContext, compilation_parameters::CompilationConfig,
        forward_pass::ArrayId, kernel::KernelDataType,
    },
    config::TransformerLayerConfig,
    parameters::ParameterTree,
};

pub struct ClassifierLayer {
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    layer_index: usize,
    copy_main_to_shortcut_mixer: Box<dyn EncodableBlock>,
    pre_attention_norm: Option<Box<dyn EncodableBlock>>,
    qkv_projection: Box<dyn EncodableBlock>,
    qk_norm: Option<Box<dyn EncodableBlock>>,
    rope: Rc<Box<dyn EncodableBlock>>,
    attention: Box<dyn EncodableBlock>,
    out_projection: Box<dyn EncodableBlock>,
    post_attention_norm: Option<Box<dyn EncodableBlock>>,
    mixer_residual_add: Box<dyn EncodableBlock>,
    copy_main_to_shortcut_mlp: Box<dyn EncodableBlock>,
    pre_mlp_norm: Box<dyn EncodableBlock>,
    mlp: Box<dyn EncodableBlock>,
    post_mlp_norm: Option<Box<dyn EncodableBlock>>,
    mlp_residual_add: Box<dyn EncodableBlock>,
}

impl ClassifierLayer {
    pub fn new(
        mtl_context: Rc<MTLContext>,
        layer_config: &TransformerLayerConfig,
        _compilation_config: Rc<CompilationConfig>,
        layer_index: usize,
        model_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        attention_scale: Option<f32>,
        layer_loader: &ParameterTree<Rc<MTLContext>>,
        rope: Rc<Box<dyn EncodableBlock>>,
    ) -> Self {
        autoreleasepool(|_| {
            let ctx = &*mtl_context; // Reference for functions expecting &MTLContext
            let attention_config = layer_config
                .mixer_config
                .as_attention()
                .expect("Classifier layers must use attention");
            let intermediate_data_type: DataType = attention_config
                .qkv_projection_config
                .activation_precision()
                .into();
            let kernel_data_type: KernelDataType =
                intermediate_data_type.into();

            let copy_main_to_shortcut_mixer: Box<dyn EncodableBlock> = Box::new(
                TensorCopy::new(
                    ctx,
                    kernel_data_type,
                    vec![ArrayId::Main, ArrayId::Shortcut].into_boxed_slice(),
                )
                .unwrap(),
            );

            let pre_attention_norm: Option<Box<dyn EncodableBlock>> =
                if let Some(norm_config) =
                    &layer_config.pre_attention_norm_config
                {
                    if layer_loader.subtree("pre_mixer_norm").is_ok() {
                        Some(Box::new(
                            Normalization::new(
                                ctx,
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
                &attention_config.qkv_projection_config,
                attention_config.has_qkv_biases,
                model_dim,
                [
                    num_heads * head_dim,
                    num_groups * head_dim,
                    num_groups * head_dim,
                ],
                ctx,
                &layer_loader.subtree("mixer.qkv_projection").unwrap(),
                ArrayId::Main,
                ArrayId::QKV,
            )
            .expect("Failed to create qkv projection");

            let qk_norm: Option<Box<dyn EncodableBlock>> = if attention_config
                .query_norm_config
                .is_some()
                || attention_config.key_norm_config.is_some()
            {
                match QKNorm::new(
                    ctx,
                    intermediate_data_type,
                    attention_config.query_norm_config.clone(),
                    attention_config.key_norm_config.clone(),
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
                &attention_config.out_projection_config,
                attention_config.has_out_biases,
                num_heads * head_dim,
                [model_dim],
                ctx,
                &layer_loader.subtree("mixer.out_projection").unwrap(),
                ArrayId::AttentionOutput,
                ArrayId::Main,
            )
            .expect("Failed to create out projection");

            let post_attention_norm: Option<Box<dyn EncodableBlock>> =
                if let Some(norm_config) =
                    &layer_config.post_attention_norm_config
                {
                    Some(Box::new(
                        Normalization::new(
                            ctx,
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

            let mixer_residual_add: Box<dyn EncodableBlock> = Box::new(
                TensorAddSwap::new(
                    ctx,
                    kernel_data_type,
                    vec![ArrayId::Shortcut, ArrayId::Main].into_boxed_slice(),
                )
                .unwrap(),
            );

            let copy_main_to_shortcut_mlp: Box<dyn EncodableBlock> = Box::new(
                TensorCopy::new(
                    ctx,
                    kernel_data_type,
                    vec![ArrayId::Main, ArrayId::Shortcut].into_boxed_slice(),
                )
                .unwrap(),
            );

            let pre_mlp_norm: Box<dyn EncodableBlock> = Box::new(
                Normalization::new(
                    ctx,
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
                &mtl_context,
                &layer_loader.subtree("mlp").unwrap(),
            )
            .expect("Failed to create mlp block");

            let post_mlp_norm: Option<Box<dyn EncodableBlock>> =
                if let Some(norm_config) = &layer_config.post_mlp_norm_config {
                    Some(Box::new(
                        Normalization::new(
                            ctx,
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

            let attention: Box<dyn EncodableBlock> = Box::new(
                Attention::new(
                    ctx,
                    kernel_data_type,
                    layer_index,
                    attention_scale,
                    attention_config.has_sinks,
                    false,
                    attention_config.sliding_window_size,
                )
                .expect("Failed to create attention kernel"),
            );

            let mlp_residual_add: Box<dyn EncodableBlock> = Box::new(
                TensorAddSwap::new(
                    ctx,
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

impl EncodableBlock for ClassifierLayer {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBufferRef,
        parameters: &EncodingParameters,
    ) {
        #[cfg(not(feature = "tracing"))]
        {
            if self.supports_shared_encoder() {
                let encoder = command_buffer.new_compute_command_encoder();
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

        let _ = parameters;
    }

    fn supports_shared_encoder(&self) -> bool {
        #[cfg(feature = "tracing")]
        {
            false
        }

        #[cfg(not(feature = "tracing"))]
        {
            self.copy_main_to_shortcut_mixer.supports_shared_encoder()
                && self
                    .pre_attention_norm
                    .as_ref()
                    .map(|b| b.supports_shared_encoder())
                    .unwrap_or(true)
                && self.qkv_projection.supports_shared_encoder()
                && self
                    .qk_norm
                    .as_ref()
                    .map(|b| b.supports_shared_encoder())
                    .unwrap_or(true)
                && self.rope.supports_shared_encoder()
                && self.attention.supports_shared_encoder()
                && self.out_projection.supports_shared_encoder()
                && self
                    .post_attention_norm
                    .as_ref()
                    .map(|b| b.supports_shared_encoder())
                    .unwrap_or(true)
                && self.mixer_residual_add.supports_shared_encoder()
                && self.copy_main_to_shortcut_mlp.supports_shared_encoder()
                && self.pre_mlp_norm.supports_shared_encoder()
                && self.mlp.supports_shared_encoder()
                && self
                    .post_mlp_norm
                    .as_ref()
                    .map(|b| b.supports_shared_encoder())
                    .unwrap_or(true)
                && self.mlp_residual_add.supports_shared_encoder()
        }
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
        parameters: &EncodingParameters,
    ) {
        debug_assert!(
            self.supports_shared_encoder(),
            "encode_with_shared_encoder called on unsupported ClassifierLayer"
        );

        self.copy_main_to_shortcut_mixer
            .encode_with_shared_encoder(state, encoder, parameters);

        if let Some(ref pre_attn_norm) = self.pre_attention_norm {
            pre_attn_norm
                .encode_with_shared_encoder(state, encoder, parameters);
        }

        self.qkv_projection
            .encode_with_shared_encoder(state, encoder, parameters);
        if let Some(ref qk_norm) = self.qk_norm {
            qk_norm.encode_with_shared_encoder(state, encoder, parameters);
        }
        self.rope.encode_with_shared_encoder(state, encoder, parameters);
        self.attention.encode_with_shared_encoder(state, encoder, parameters);
        self.out_projection
            .encode_with_shared_encoder(state, encoder, parameters);

        if let Some(ref post_attn_norm) = self.post_attention_norm {
            post_attn_norm
                .encode_with_shared_encoder(state, encoder, parameters);
        }

        self.mixer_residual_add
            .encode_with_shared_encoder(state, encoder, parameters);

        self.copy_main_to_shortcut_mlp
            .encode_with_shared_encoder(state, encoder, parameters);

        self.pre_mlp_norm
            .encode_with_shared_encoder(state, encoder, parameters);
        self.mlp.encode_with_shared_encoder(state, encoder, parameters);

        if let Some(ref post_mlp_norm) = self.post_mlp_norm {
            post_mlp_norm
                .encode_with_shared_encoder(state, encoder, parameters);
        }

        self.mlp_residual_add
            .encode_with_shared_encoder(state, encoder, parameters);
    }
}
