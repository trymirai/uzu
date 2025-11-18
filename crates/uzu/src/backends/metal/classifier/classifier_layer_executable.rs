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
            TensorAddSwap, TensorCopy,
        },
    },
    config::TransformerLayerConfig,
    parameters::ParameterTree,
};

pub struct ClassifierLayerExecutable {
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    layer_index: usize,
    copy_main_to_shortcut: Box<dyn EncodableWithState>,
    pre_attention_norm: Option<Box<dyn EncodableWithState>>,
    qkv_projection: Box<dyn EncodableWithState>,
    rope: Rc<Box<dyn EncodableWithState>>,
    attention: Box<dyn EncodableWithState>,
    out_projection: Box<dyn EncodableWithState>,
    main_shortcut_add_swap: Box<dyn EncodableWithState>,
    pre_mlp_norm: Box<dyn EncodableWithState>,
    mlp: Box<dyn EncodableWithState>,
    main_shortcut_add_swap_2: Box<dyn EncodableWithState>,
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

            let copy_main_to_shortcut: Box<dyn EncodableWithState> = Box::new(
                TensorCopy::new(
                    mtl_context,
                    kernel_data_type,
                    vec![ArrayId::Main, ArrayId::Shortcut].into_boxed_slice(),
                )
                .unwrap(),
            );

            let pre_attention_norm: Option<Box<dyn EncodableWithState>> =
                if let Some(norm_config) =
                    &layer_config.pre_attention_norm_config
                {
                    if layer_loader.subtree("pre_attention_norm").is_ok() {
                        Some(Box::new(
                            NormalizationEncodable::new(
                                mtl_context,
                                intermediate_data_type,
                                norm_config.clone(),
                                ArrayId::Main,
                                ArrayId::Main,
                                &layer_loader
                                    .subtree("pre_attention_norm")
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
                &layer_loader.subtree("attention.qkv_projection").unwrap(),
                ArrayId::Main,
                ArrayId::QKV,
                &compilation_config.descriptor_mlp,
            );

            let out_projection = transformer_layer::linear_block(
                &layer_config.attention_config.out_projection_config,
                layer_config.attention_config.has_out_biases,
                num_heads * head_dim,
                [model_dim],
                mtl_context,
                &layer_loader.subtree("attention.out_projection").unwrap(),
                ArrayId::AttentionOutput,
                ArrayId::Main,
                &compilation_config.descriptor_mlp,
            );

            let main_shortcut_add_swap: Box<dyn EncodableWithState> = Box::new(
                TensorAddSwap::new(
                    mtl_context,
                    kernel_data_type,
                    vec![ArrayId::Main, ArrayId::Shortcut].into_boxed_slice(),
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

            let main_shortcut_add_swap_2: Box<dyn EncodableWithState> =
                Box::new(
                    TensorAddSwap::new(
                        mtl_context,
                        kernel_data_type,
                        vec![ArrayId::Main, ArrayId::Shortcut]
                            .into_boxed_slice(),
                    )
                    .unwrap(),
                );

            let attention: Box<dyn EncodableWithState> = Box::new(
                AttentionKernelEncodable::new(
                    mtl_context,
                    kernel_data_type,
                    layer_index,
                    attention_scale,
                    layer_config.attention_config.has_sinks,
                    false, // is_causal - Classifier uses bidirectional attention
                    layer_config.sliding_window_size, // Pass sliding window size
                )
                .expect("Failed to create attention kernel"),
            );

            Self {
                layer_index,
                copy_main_to_shortcut,
                pre_attention_norm,
                qkv_projection,
                rope,
                attention,
                out_projection,
                main_shortcut_add_swap,
                pre_mlp_norm,
                mlp,
                main_shortcut_add_swap_2,
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

        self.copy_main_to_shortcut.encode(state, command_buffer, parameters);

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

        self.main_shortcut_add_swap.encode(state, command_buffer, parameters);
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

        self.main_shortcut_add_swap_2.encode(state, command_buffer, parameters);
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
