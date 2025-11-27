use std::rc::Rc;

use metal::ComputeCommandEncoderRef;
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::rc::autoreleasepool;

use crate::{
    DataType,
    backends::metal::{
        MTLContext,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ArrayId, ForwardPassState, FrozenState,
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

impl LayerExecutables {
    fn encode_with_shared_encoder_impl(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
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

        self.copy_main_to_shortcut
            .encode_with_shared_encoder(state, encoder, parameters);

        self.pre_attention_norm
            .encode_with_shared_encoder(state, encoder, parameters);
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
                mixer.encode_with_shared_encoder(state, encoder, parameters);
                if let Some(layer_traces) = layer_traces.clone() {
                    state.copy_array(
                        ArrayId::Main,
                        layer_traces.borrow().attention.clone(),
                    );
                }
            },
        }

        if let Some(post_attention_norm) = &self.post_attention_norm {
            post_attention_norm
                .encode_with_shared_encoder(state, encoder, parameters);
            if let Some(layer_traces) = layer_traces.clone() {
                state.copy_array(
                    ArrayId::Main,
                    layer_traces.borrow().post_attention_norm.clone(),
                );
            }
        }

        self.main_shortcut_add_swap
            .encode_with_shared_encoder(state, encoder, parameters);
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(
                ArrayId::Main,
                layer_traces.borrow().mlp_inputs.clone(),
            );
        }

        self.pre_mlp_norm
            .encode_with_shared_encoder(state, encoder, parameters);
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(
                ArrayId::Main,
                layer_traces.borrow().pre_mlp_norm.clone(),
            );
        }

        self.mlp.encode_with_shared_encoder(state, encoder, parameters);
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(ArrayId::Main, layer_traces.borrow().mlp.clone());
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            post_mlp_norm
                .encode_with_shared_encoder(state, encoder, parameters);
            if let Some(layer_traces) = layer_traces.clone() {
                state.copy_array(
                    ArrayId::Main,
                    layer_traces.borrow().post_mlp_norm.clone(),
                );
            }
        }

        self.main_shortcut_add_swap
            .encode_with_shared_encoder(state, encoder, parameters);
        if let Some(layer_traces) = layer_traces.clone() {
            state.copy_array(
                ArrayId::Main,
                layer_traces.borrow().outputs.clone(),
            );
        }
    }
}

impl EncodableWithState for LayerExecutables {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let encoder = mtl_command_buffer.new_compute_command_encoder();

        // GPU wait on previous encoder's fence
        if let Some(prev_fence) = state.fence_registry.take_previous() {
            encoder.wait_for_fence(&prev_fence);
        }

        self.encode_with_shared_encoder_impl(state, encoder, parameters);

        // GPU signal fence for next encoder
        let fence = state.fence_registry.new_fence();
        encoder.update_fence(&fence);
        encoder.end_encoding();
        state.fence_registry.set_current(fence);

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
        parameters: &EncodingParameters,
    ) {
        self.encode_with_shared_encoder_impl(state, encoder, parameters);
    }
}

impl LayerExecutables {
    /// Check if this layer supports parallel encoding
    pub fn supports_parallel_encode(&self) -> bool {
        // Check if all sub-components support parallel encoding
        let mixer_supports = match &self.mixer {
            MixerExecutables::Attention {
                ..
            } => false, // Not yet implemented
            MixerExecutables::StateSpace {
                mixer,
            } => mixer.supports_parallel_encode(),
        };

        self.copy_main_to_shortcut.supports_parallel_encode()
            && self.pre_attention_norm.supports_parallel_encode()
            && mixer_supports
            && self.main_shortcut_add_swap.supports_parallel_encode()
            && self.pre_mlp_norm.supports_parallel_encode()
            && self.mlp.supports_parallel_encode()
    }

    /// Encode directly using FrozenState (for testing/validation)
    pub fn encode_parallel_direct(
        &self,
        encoder: &ComputeCommandEncoderRef,
        frozen: &FrozenState,
        parameters: &EncodingParameters,
    ) {
        // Copy main to shortcut
        self.copy_main_to_shortcut.encode_parallel(encoder, frozen, parameters);

        // Pre-attention norm
        self.pre_attention_norm.encode_parallel(encoder, frozen, parameters);

        // Mixer
        match &self.mixer {
            MixerExecutables::Attention {
                qkv_projection,
                qk_norm,
                rope,
                attention,
                out_projection,
            } => {
                // Attention components - all must support parallel encode
                assert!(
                    qkv_projection.supports_parallel_encode(),
                    "qkv_projection must support parallel encode"
                );
                qkv_projection.encode_parallel(encoder, frozen, parameters);

                if let Some(norm) = qk_norm {
                    assert!(
                        norm.supports_parallel_encode(),
                        "qk_norm must support parallel encode"
                    );
                    norm.encode_parallel(encoder, frozen, parameters);
                }

                assert!(
                    rope.supports_parallel_encode(),
                    "rope must support parallel encode"
                );
                rope.encode_parallel(encoder, frozen, parameters);

                assert!(
                    attention.supports_parallel_encode(),
                    "attention must support parallel encode"
                );
                attention.encode_parallel(encoder, frozen, parameters);

                assert!(
                    out_projection.supports_parallel_encode(),
                    "out_projection must support parallel encode"
                );
                out_projection.encode_parallel(encoder, frozen, parameters);
            },
            MixerExecutables::StateSpace {
                mixer,
            } => {
                mixer.encode_parallel(encoder, frozen, parameters);
            },
        }

        // Post-attention norm (if present)
        if let Some(post_attention_norm) = &self.post_attention_norm {
            if post_attention_norm.supports_parallel_encode() {
                post_attention_norm
                    .encode_parallel(encoder, frozen, parameters);
            }
        }

        // Add-swap (first)
        self.main_shortcut_add_swap
            .encode_parallel(encoder, frozen, parameters);

        // Pre-MLP norm
        self.pre_mlp_norm.encode_parallel(encoder, frozen, parameters);

        // MLP
        self.mlp.encode_parallel(encoder, frozen, parameters);

        // Post-MLP norm (if present)
        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            if post_mlp_norm.supports_parallel_encode() {
                post_mlp_norm.encode_parallel(encoder, frozen, parameters);
            }
        }

        // Add-swap (second)
        self.main_shortcut_add_swap
            .encode_parallel(encoder, frozen, parameters);
    }

    /// Create a thread-safe encoder that can be sent to another thread
    pub fn create_parallel_encoder(&self) -> ParallelLayerEncoder {
        ParallelLayerEncoder::new(self)
    }
}

/// Thread-safe layer encoder that can be sent to worker threads.
/// Captures raw pointers to Metal pipeline states (which are thread-safe).
pub struct ParallelLayerEncoder {
    layer_index: usize,
    // Store raw pointers to the layer's sub-components
    // SAFETY: Metal pipeline states are thread-safe GPU objects
    layer_ptr: *const LayerExecutables,
}

// SAFETY: Metal pipelines are thread-safe GPU objects.
// The layer pointer is valid for the lifetime of the encoding operation.
unsafe impl Send for ParallelLayerEncoder {}
unsafe impl Sync for ParallelLayerEncoder {}

impl ParallelLayerEncoder {
    fn new(layer: &LayerExecutables) -> Self {
        Self {
            layer_index: layer.layer_index,
            layer_ptr: layer as *const LayerExecutables,
        }
    }

    /// Encode the layer using frozen state
    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        frozen: &FrozenState,
        parameters: &EncodingParameters,
    ) {
        // SAFETY: The layer pointer is valid because:
        // 1. ParallelEncodingContext::encode_and_commit runs in a rayon scope
        // 2. The scope blocks until all tasks complete
        // 3. The layers (in DecoderExecutables) outlive the encoding operation
        let layer = unsafe { &*self.layer_ptr };
        layer.encode_parallel_direct(encoder, frozen, parameters);
    }
}
