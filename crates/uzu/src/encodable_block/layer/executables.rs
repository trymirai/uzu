//! Layer executables - a single decoder layer with mixer, norms, and MLP.

use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use super::MixerExecutables;
use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
        kernel::{
            Kernels, RMSNormCopyHadamardMulKernel, RMSNormCopyKernel, ResidualAddRMSNormHadamardMulKernel,
            ResidualAddRMSNormKernel,
        },
    },
    config::{DecoderLayerConfig, DecoderLayerType, MixerConfig, NormalizationConfig, UpcastMode},
    encodable_block::{
        Attention, EncodingParameters, Linear, MambaMixer, Mlp, QKNorm, RMSNorm, RMSNormHadamard, Rope, ShortConvMixer,
        TensorAddSwap, TensorCopy,
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub enum NormBlock<B: Backend> {
    Plain(RMSNorm<B>),
    FusedHadamard(RMSNormHadamard<B>),
}

impl<B: Backend> NormBlock<B> {
    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        match self {
            Self::Plain(norm) => norm.encode(state, encoder),
            Self::FusedHadamard(norm) => norm.encode(state, encoder),
        }
    }
}

enum FusedCopyNorm<B: Backend> {
    Plain {
        kernel: <B::Kernels as Kernels>::RMSNormCopyKernel,
        config: NormalizationConfig,
        scales_buffer: Rc<RefCell<B::Buffer>>,
    },
    Hadamard {
        kernel: <B::Kernels as Kernels>::RMSNormCopyHadamardMulKernel,
        config: NormalizationConfig,
        scales_buffer: Rc<RefCell<B::Buffer>>,
        hadamard_factors: Rc<RefCell<B::Buffer>>,
    },
}

impl<B: Backend> FusedCopyNorm<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let main_array = state.array(ArrayId::Main);
        let shortcut_array = state.array(ArrayId::Shortcut);
        let batch_len = main_array.shape()[0].min(state.active_suffix_length()) as u32;
        let element_count = main_array.shape()[1] as u32;
        let main_buf_rc = main_array.buffer();
        let shortcut_buf_rc = shortcut_array.buffer();
        let mut main_borrow = main_buf_rc.borrow_mut();
        let mut shortcut_borrow = shortcut_buf_rc.borrow_mut();

        match self {
            Self::Plain {
                kernel,
                config,
                scales_buffer,
            } => {
                kernel.encode(
                    (main_borrow.deref_mut(), 0),
                    (shortcut_borrow.deref_mut(), 0),
                    scales_buffer.borrow().deref(),
                    batch_len,
                    element_count,
                    config.epsilon,
                    config.scale_offset.unwrap_or(0.0),
                    config.upcast_mode == UpcastMode::FullLayer,
                    encoder,
                );
            },
            Self::Hadamard {
                kernel,
                config,
                scales_buffer,
                hadamard_factors,
            } => {
                kernel.encode(
                    (main_borrow.deref_mut(), 0),
                    (shortcut_borrow.deref_mut(), 0),
                    scales_buffer.borrow().deref(),
                    hadamard_factors.borrow().deref(),
                    batch_len,
                    element_count,
                    config.epsilon,
                    config.scale_offset.unwrap_or(0.0),
                    config.upcast_mode == UpcastMode::FullLayer,
                    encoder,
                );
            },
        }
        Ok(())
    }
}

enum FusedResidualAddNorm<B: Backend> {
    Plain {
        kernel: <B::Kernels as Kernels>::ResidualAddRMSNormKernel,
        config: NormalizationConfig,
        scales_buffer: Rc<RefCell<B::Buffer>>,
    },
    Hadamard {
        kernel: <B::Kernels as Kernels>::ResidualAddRMSNormHadamardMulKernel,
        config: NormalizationConfig,
        scales_buffer: Rc<RefCell<B::Buffer>>,
        hadamard_factors: Rc<RefCell<B::Buffer>>,
    },
}

impl<B: Backend> FusedResidualAddNorm<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let main_array = state.array(ArrayId::Main);
        let shortcut_array = state.array(ArrayId::Shortcut);
        let batch_len = main_array.shape()[0].min(state.active_suffix_length()) as u32;
        let element_count = main_array.shape()[1] as u32;
        let main_buf_rc = main_array.buffer();
        let shortcut_buf_rc = shortcut_array.buffer();
        let mut main_borrow = main_buf_rc.borrow_mut();
        let mut shortcut_borrow = shortcut_buf_rc.borrow_mut();

        match self {
            Self::Plain {
                kernel,
                config,
                scales_buffer,
            } => {
                kernel.encode(
                    (main_borrow.deref_mut(), 0),
                    (shortcut_borrow.deref_mut(), 0),
                    scales_buffer.borrow().deref(),
                    batch_len,
                    element_count,
                    config.epsilon,
                    config.scale_offset.unwrap_or(0.0),
                    config.upcast_mode == UpcastMode::FullLayer,
                    encoder,
                );
            },
            Self::Hadamard {
                kernel,
                config,
                scales_buffer,
                hadamard_factors,
            } => {
                kernel.encode(
                    (main_borrow.deref_mut(), 0),
                    (shortcut_borrow.deref_mut(), 0),
                    scales_buffer.borrow().deref(),
                    hadamard_factors.borrow().deref(),
                    batch_len,
                    element_count,
                    config.epsilon,
                    config.scale_offset.unwrap_or(0.0),
                    config.upcast_mode == UpcastMode::FullLayer,
                    encoder,
                );
            },
        }
        Ok(())
    }
}

fn try_create_fused_copy_norm<B: Backend>(
    context: &B::Context,
    intermediate_data_type: DataType,
    config: &NormalizationConfig,
    norm_tree: &ParameterTree<B::Context>,
    hadamard_factors: Option<Rc<RefCell<B::Buffer>>>,
) -> Option<FusedCopyNorm<B>> {
    let scales = norm_tree.leaf_array("scales").ok()?;
    let scale_data_type: DataType = config.scale_precision.into();
    let accumulation_data_type: DataType = config.accumulation_precision.into();

    if let Some(factors) = hadamard_factors {
        let kernel = <B::Kernels as Kernels>::RMSNormCopyHadamardMulKernel::new(
            context,
            scale_data_type,
            intermediate_data_type,
            accumulation_data_type,
        )
        .ok()?;
        Some(FusedCopyNorm::Hadamard {
            kernel,
            config: config.clone(),
            scales_buffer: scales.buffer(),
            hadamard_factors: factors,
        })
    } else {
        let kernel = <B::Kernels as Kernels>::RMSNormCopyKernel::new(
            context,
            scale_data_type,
            intermediate_data_type,
            accumulation_data_type,
        )
        .ok()?;
        Some(FusedCopyNorm::Plain {
            kernel,
            config: config.clone(),
            scales_buffer: scales.buffer(),
        })
    }
}

fn try_create_fused_residual_add_norm<B: Backend>(
    context: &B::Context,
    intermediate_data_type: DataType,
    config: &NormalizationConfig,
    norm_tree: &ParameterTree<B::Context>,
    hadamard_factors: Option<Rc<RefCell<B::Buffer>>>,
) -> Option<FusedResidualAddNorm<B>> {
    let scales = norm_tree.leaf_array("scales").ok()?;
    let scale_data_type: DataType = config.scale_precision.into();
    let accumulation_data_type: DataType = config.accumulation_precision.into();

    if let Some(factors) = hadamard_factors {
        let kernel = <B::Kernels as Kernels>::ResidualAddRMSNormHadamardMulKernel::new(
            context,
            scale_data_type,
            intermediate_data_type,
            accumulation_data_type,
        )
        .ok()?;
        Some(FusedResidualAddNorm::Hadamard {
            kernel,
            config: config.clone(),
            scales_buffer: scales.buffer(),
            hadamard_factors: factors,
        })
    } else {
        let kernel = <B::Kernels as Kernels>::ResidualAddRMSNormKernel::new(
            context,
            scale_data_type,
            intermediate_data_type,
            accumulation_data_type,
        )
        .ok()?;
        Some(FusedResidualAddNorm::Plain {
            kernel,
            config: config.clone(),
            scales_buffer: scales.buffer(),
        })
    }
}

/// A single decoder layer with all its components.
pub struct LayerExecutables<B: Backend> {
    pub layer_index: usize,
    pub copy_main_to_shortcut: TensorCopy<B>,
    pub pre_attention_norm: NormBlock<B>,
    pub(crate) mixer: MixerExecutables<B>,
    pub post_attention_norm: Option<RMSNorm<B>>,
    pub main_shortcut_add_swap: TensorAddSwap<B>,
    pub pre_mlp_norm: NormBlock<B>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<RMSNorm<B>>,
    fused_pre_attn: Option<FusedCopyNorm<B>>,
    fused_pre_mlp: Option<FusedResidualAddNorm<B>>,
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
        let copy_main_to_shortcut =
            TensorCopy::<B>::new(context, intermediate_data_type, ArrayId::Main, ArrayId::Shortcut).unwrap();

        let norm_tree = decoder_layer_loader.subtree("pre_mixer_norm").unwrap();

        let (pre_attention_norm, mixer, fused_pre_attn) = match &layer_config.mixer_config {
            MixerConfig::Attention(attention_config) => {
                let rope_block = rope.expect("RoPE encoder missing for attention layer");

                let (qkv_projection, input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                    &attention_config.qkv_projection_config,
                    attention_config.has_qkv_biases,
                    model_dim,
                    [num_heads * head_dim, num_groups * head_dim, num_groups * head_dim],
                    context,
                    &decoder_layer_loader.subtree("mixer.qkv_projection").unwrap(),
                    ArrayId::Main,
                    ArrayId::QKV,
                )
                .expect("Failed to create qkv projection");

                let fused_pre_attn = try_create_fused_copy_norm(
                    context,
                    intermediate_data_type,
                    &layer_config.pre_attention_norm_config,
                    &norm_tree,
                    input_hadamard_factors.clone(),
                );

                let pre_attention_norm = if let Some(factors) = input_hadamard_factors {
                    NormBlock::FusedHadamard(
                        RMSNormHadamard::new(
                            context,
                            intermediate_data_type,
                            layer_config.pre_attention_norm_config.clone(),
                            ArrayId::Main,
                            ArrayId::Main,
                            &norm_tree,
                            factors,
                        )
                        .expect("Failed to create fused RMSNorm+Hadamard kernel"),
                    )
                } else {
                    NormBlock::Plain(
                        RMSNorm::new(
                            context,
                            intermediate_data_type,
                            layer_config.pre_attention_norm_config.clone(),
                            ArrayId::Main,
                            ArrayId::Main,
                            &norm_tree,
                        )
                        .expect("Failed to create RMS norm kernel"),
                    )
                };

                let gate_projection = if attention_config.has_gate {
                    Some(
                        <dyn Linear<B>>::new(
                            &attention_config.qkv_projection_config,
                            false,
                            model_dim,
                            [num_heads * head_dim],
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
                            num_heads,
                            num_groups,
                            head_dim,
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
                    num_heads * head_dim,
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
                    attention_config.has_gate,
                )
                .expect("Failed to create AttentionWrapper kernel");

                (
                    pre_attention_norm,
                    MixerExecutables::Attention {
                        qkv_projection,
                        gate_projection,
                        qk_norm,
                        rope: rope_block,
                        attention,
                        out_projection,
                    },
                    fused_pre_attn,
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
                let norm = NormBlock::Plain(
                    RMSNorm::new(
                        context,
                        intermediate_data_type,
                        layer_config.pre_attention_norm_config.clone(),
                        ArrayId::Main,
                        ArrayId::Main,
                        &norm_tree,
                    )
                    .expect("Failed to create RMS norm kernel"),
                );
                let fused_pre_attn = try_create_fused_copy_norm(
                    context,
                    intermediate_data_type,
                    &layer_config.pre_attention_norm_config,
                    &norm_tree,
                    None,
                );
                (
                    norm,
                    MixerExecutables::StateSpace {
                        mixer,
                    },
                    fused_pre_attn,
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
                let fused_pre_attn = try_create_fused_copy_norm(
                    context,
                    intermediate_data_type,
                    &layer_config.pre_attention_norm_config,
                    &norm_tree,
                    input_hadamard_factors.clone(),
                );
                let norm = if let Some(factors) = input_hadamard_factors {
                    NormBlock::FusedHadamard(
                        RMSNormHadamard::new(
                            context,
                            intermediate_data_type,
                            layer_config.pre_attention_norm_config.clone(),
                            ArrayId::Main,
                            ArrayId::Main,
                            &norm_tree,
                            factors,
                        )
                        .expect("Failed to create fused RMSNorm+Hadamard kernel"),
                    )
                } else {
                    NormBlock::Plain(
                        RMSNorm::new(
                            context,
                            intermediate_data_type,
                            layer_config.pre_attention_norm_config.clone(),
                            ArrayId::Main,
                            ArrayId::Main,
                            &norm_tree,
                        )
                        .expect("Failed to create RMS norm kernel"),
                    )
                };
                (
                    norm,
                    MixerExecutables::ShortConv {
                        mixer,
                    },
                    fused_pre_attn,
                )
            },
            MixerConfig::DeltaNet(_) => unimplemented!("DeltaNet mixer"),
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
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
        };

        let main_shortcut_add_swap =
            TensorAddSwap::<B>::new(context, intermediate_data_type, ArrayId::Shortcut, ArrayId::Main).unwrap();

        let mlp_norm_tree = decoder_layer_loader.subtree("pre_mlp_norm").unwrap();

        let (mlp, mlp_input_hadamard_factors) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            model_dim,
            hidden_dim,
            context,
            &decoder_layer_loader.subtree("mlp").unwrap(),
        )
        .expect("Failed to create mlp block");

        let fused_pre_mlp = try_create_fused_residual_add_norm(
            context,
            intermediate_data_type,
            &layer_config.pre_mlp_norm_config,
            &mlp_norm_tree,
            mlp_input_hadamard_factors.clone(),
        );

        let pre_mlp_norm = if let Some(factors) = mlp_input_hadamard_factors {
            NormBlock::FusedHadamard(
                RMSNormHadamard::new(
                    context,
                    intermediate_data_type,
                    layer_config.pre_mlp_norm_config.clone(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &mlp_norm_tree,
                    factors,
                )
                .expect("Failed to create fused RMSNorm+Hadamard kernel"),
            )
        } else {
            NormBlock::Plain(
                RMSNorm::new(
                    context,
                    intermediate_data_type,
                    layer_config.pre_mlp_norm_config.clone(),
                    ArrayId::Main,
                    ArrayId::Main,
                    &mlp_norm_tree,
                )
                .expect("Failed to create RMS norm kernel"),
            )
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
            fused_pre_attn,
            fused_pre_mlp,
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

        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().inputs.clone());
        }

        if self.layer_index == 0 {
            if let Some(ref fused) = self.fused_pre_attn {
                fused.encode(state, encoder)?;
            } else {
                self.copy_main_to_shortcut.encode(state, encoder)?;
                self.pre_attention_norm.encode(state, encoder)?;
            }
        } else {
            // For layers > 0 the previous layer's TensorAddSwap already wrote the
            // identical value to both Main and Shortcut, so the copy is a no-op.
            self.pre_attention_norm.encode(state, encoder)?;
        }
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().pre_attention_norm.clone());
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
                if let Some(norm) = qk_norm {
                    norm.encode(state, encoder)?;
                }
                rope.encode(state, encoder)?;
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
        }

        if let Some(post_attention_norm) = &self.post_attention_norm {
            post_attention_norm.encode(state, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().post_attention_norm.clone());
            }
        }
        //main = attention_result

        if let Some(ref fused) = self.fused_pre_mlp {
            fused.encode(state, encoder)?;
        } else {
            self.main_shortcut_add_swap.encode(state, encoder)?;
            // shortcut = input + attention_result
            // main = input + attention_result
            self.pre_mlp_norm.encode(state, encoder)?;
        }
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().mlp_inputs.clone());
        }
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
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

        self.main_shortcut_add_swap.encode(state, encoder)?;
        // shortcut = input + attention_result + mlp_result
        // main = input + attention_result + mlp_result
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            state.encode_copy_array(encoder, ArrayId::Main, layer_traces.borrow().outputs.clone());
        }

        Ok(())
    }
}
