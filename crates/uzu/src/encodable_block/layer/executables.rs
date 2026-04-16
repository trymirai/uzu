//! Layer executables - a single decoder layer with mixer, norms, and MLP.

use std::rc::Rc;
#[cfg(feature = "tracing")]
use std::{cell::RefCell, ops::DerefMut};

use super::MixerExecutables;
#[cfg(feature = "tracing")]
use crate::backends::common::{Kernels, kernel::TensorAddBiasKernel};
use crate::{
    DataType,
    backends::common::{Allocation, Backend, Encoder},
    config::{DecoderLayerConfig, DecoderLayerType, MixerConfig},
    encodable_block::{
        Attention, AttentionArguments, DeltaNetArguments, DeltaNetMixer, EncodingParameters, Linear, MambaArguments,
        MambaMixer, Mlp, QKNorm, RMSNorm, Rope, ShortConvArguments, ShortConvMixer,
    },
    forward_pass::{cache_layers::CacheLayer, state::RopeType},
    parameters::ParameterTree,
};

/// A single decoder layer with all its components.
pub struct LayerExecutables<B: Backend> {
    pub layer_index: usize,
    #[cfg(feature = "tracing")]
    pub tensor_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    pub pre_attention_norm: RMSNorm<B>,
    pub(crate) mixer: MixerExecutables<B>,
    pub post_attention_norm: Option<RMSNorm<B>>,
    pub pre_mlp_norm: RMSNorm<B>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<RMSNorm<B>>,
    #[cfg(feature = "tracing")]
    model_dim: usize,
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

        let (mixer, mixer_hadamard_factors) = match &layer_config.mixer_config {
            MixerConfig::Attention(attention_config) => {
                let rope_block = rope.expect("RoPE encoder missing for attention layer");

                let layer_num_heads = attention_config.num_heads.unwrap_or(num_heads);
                let layer_num_groups = attention_config.num_groups.unwrap_or(num_groups);
                let layer_head_dim = attention_config.head_dim.unwrap_or(head_dim);

                let q_dim = layer_num_heads * layer_head_dim;
                let kv_dim = layer_num_groups * layer_head_dim;

                let (qkv_projection, input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                    &attention_config.qkv_projection_config,
                    attention_config.has_qkv_biases,
                    model_dim,
                    [q_dim, kv_dim, kv_dim],
                    context,
                    &decoder_layer_loader.subtree("mixer.qkv_projection").unwrap(),
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
                )
                .expect("Failed to create out projection");

                let attention = Attention::new(
                    context,
                    intermediate_data_type,
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
                        attention,
                        out_projection,
                        num_heads: layer_num_heads,
                        num_groups: layer_num_groups,
                        head_dim: layer_head_dim,
                    },
                    input_hadamard_factors,
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
                )
            },
        };

        let pre_attention_norm = RMSNorm::new(
            context,
            intermediate_data_type,
            layer_config.pre_attention_norm_config.clone(),
            &decoder_layer_loader.subtree("pre_mixer_norm").unwrap(),
            mixer_hadamard_factors,
            true,
            layer_index > 0,
        )
        .expect("Failed to create RMS norm kernel");

        let post_attention_norm = if let Some(norm_config) = &layer_config.post_attention_norm_config {
            Some(
                RMSNorm::new(
                    context,
                    intermediate_data_type,
                    norm_config.clone(),
                    &decoder_layer_loader.subtree("post_mixer_norm").unwrap(),
                    None,
                    false,
                    false,
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
        };

        let (mlp, mlp_input_hadamard_factors) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            model_dim,
            hidden_dim,
            context,
            &decoder_layer_loader.subtree("mlp").unwrap(),
        )
        .expect("Failed to create mlp block");

        let pre_mlp_norm = RMSNorm::new(
            context,
            intermediate_data_type,
            layer_config.pre_mlp_norm_config.clone(),
            &decoder_layer_loader.subtree("pre_mlp_norm").unwrap(),
            mlp_input_hadamard_factors,
            true,
            true,
        )
        .expect("Failed to create RMS norm kernel");

        let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
            Some(
                RMSNorm::new(
                    context,
                    intermediate_data_type,
                    norm_config.clone(),
                    &decoder_layer_loader.subtree("post_mlp_norm").unwrap(),
                    None,
                    false,
                    false,
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
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
            #[cfg(feature = "tracing")]
            model_dim,
        }
    }

    pub fn encode(
        &self,
        args: LayerArguments<'_, B>,
        parameters: &EncodingParameters,
        main: Allocation<B>,
        shortcut: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let LayerArguments {
            context,
            batch_dim,
            token_positions,
            token_parents,
            token_subtrie_ranges,
            attention_sinks,
            rope_cosines,
            rope_sines,
            rope_max_sequence_length,
            rope_dim,
            sampling_start,
            sampling_length,
            mut cache_layer,
            #[cfg(feature = "tracing")]
            trace,
        } = args;
        #[cfg(feature = "tracing")]
        let layer_traces = trace;

        let main = self.pre_attention_norm.encode(&main, 0, batch_dim, Some(shortcut), encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            let traces = layer_traces.borrow();
            crate::backends::common::allocation_helpers::encode_copy_allocation_to_array(
                encoder,
                shortcut,
                &traces.inputs,
            );
            crate::backends::common::allocation_helpers::encode_copy_allocation_to_array(
                encoder,
                &main,
                &traces.pre_attention_norm,
            );
        }

        let mut main = match &self.mixer {
            MixerExecutables::Attention {
                qkv_projection,
                gate_projection,
                qk_norm,
                rope,
                attention,
                out_projection,
                num_heads,
                num_groups,
                head_dim,
            } => {
                let mut qkv = qkv_projection.encode(context, &main, batch_dim, encoder)?;
                let gate = if let Some(gate_proj) = gate_projection {
                    Some(gate_proj.encode(context, &main, batch_dim, encoder)?)
                } else {
                    None
                };
                if let Some(norm) = qk_norm {
                    norm.encode(&mut qkv, batch_dim, encoder)?;
                }
                let cosines = rope_cosines.expect("Attention layer requires RoPE cosine allocation");
                let sines = rope_sines.expect("Attention layer requires RoPE sine allocation");
                let (queries, rotated_keys) = rope.encode(
                    &qkv,
                    token_positions,
                    cosines,
                    sines,
                    batch_dim,
                    *num_heads,
                    *num_groups,
                    *head_dim,
                    rope_max_sequence_length,
                    rope_dim,
                    encoder,
                )?;
                let kv_cache_layer = cache_layer
                    .as_deref_mut()
                    .map(|layer| layer.as_transformer_mut().expect("Attention layer expects transformer cache"));
                let attention_output = attention.encode(
                    AttentionArguments {
                        context,
                        projection_step: parameters.projection_step.unwrap_or(0),
                        token_subtrie_ranges,
                        attention_sinks,
                        kv_cache_layer,
                    },
                    &qkv,
                    &queries,
                    rotated_keys,
                    gate.as_ref(),
                    batch_dim,
                    *num_heads,
                    *num_groups,
                    *head_dim,
                    encoder,
                )?;
                out_projection.encode(context, &attention_output, batch_dim, encoder)?
            },
            MixerExecutables::StateSpace {
                mixer,
            } => {
                let layer = cache_layer
                    .as_deref_mut()
                    .expect("State-space layer requires cache state")
                    .as_state_space_mut()
                    .expect("State-space mixer expects SSM cache layer");
                mixer.encode(
                    MambaArguments {
                        context,
                        active_row_count: batch_dim,
                        layer,
                    },
                    &main,
                    encoder,
                )?
            },
            MixerExecutables::ShortConv {
                mixer,
            } => {
                let layer = cache_layer
                    .as_deref_mut()
                    .expect("ShortConv layer requires cache state")
                    .as_short_conv_mut()
                    .expect("ShortConv mixer expects ShortConv cache layer");
                mixer.encode(
                    ShortConvArguments {
                        context,
                        active_row_count: batch_dim,
                        sampling_start,
                        sampling_length,
                        token_parents,
                        layer,
                    },
                    &main,
                    encoder,
                )?
            },
            MixerExecutables::DeltaNet {
                mixer,
            } => {
                let layer = cache_layer
                    .as_deref_mut()
                    .expect("DeltaNet layer requires cache state")
                    .as_delta_net_mut()
                    .expect("DeltaNet mixer expects DeltaNet cache layer");
                mixer.encode(
                    DeltaNetArguments {
                        context,
                        active_row_count: batch_dim,
                        layer,
                    },
                    &main,
                    encoder,
                )?
            },
        };
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            let traces = layer_traces.borrow();
            crate::backends::common::allocation_helpers::encode_copy_allocation_to_array(
                encoder,
                &main,
                &traces.attention,
            );
        }

        if let Some(post_attention_norm) = &self.post_attention_norm {
            main = post_attention_norm.encode(&main, 0, batch_dim, None, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                let traces = layer_traces.borrow();
                crate::backends::common::allocation_helpers::encode_copy_allocation_to_array(
                    encoder,
                    &main,
                    &traces.post_attention_norm,
                );
            }
        }

        main = self.pre_mlp_norm.encode(&main, 0, batch_dim, Some(shortcut), encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            let traces = layer_traces.borrow();
            crate::backends::common::allocation_helpers::encode_copy_allocation_to_array(
                encoder,
                shortcut,
                &traces.mlp_inputs,
            );
            crate::backends::common::allocation_helpers::encode_copy_allocation_to_array(
                encoder,
                &main,
                &traces.pre_mlp_norm,
            );
        }

        main = self.mlp.encode(context, &main, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            let traces = layer_traces.borrow();
            crate::backends::common::allocation_helpers::encode_copy_allocation_to_array(encoder, &main, &traces.mlp);
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            main = post_mlp_norm.encode(&main, 0, batch_dim, None, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(ref layer_traces) = layer_traces {
                let traces = layer_traces.borrow();
                crate::backends::common::allocation_helpers::encode_copy_allocation_to_array(
                    encoder,
                    &main,
                    &traces.post_mlp_norm,
                );
            }
        }

        #[cfg(feature = "tracing")]
        if let Some(ref layer_traces) = layer_traces {
            let size = (batch_dim * self.model_dim) as u32;
            let output = layer_traces.borrow().outputs.clone();
            let output_buffer = output.buffer();
            self.tensor_add.encode(
                Some(&main),
                &*shortcut,
                (output_buffer.borrow_mut().deref_mut(), output.offset()),
                size,
                size,
                encoder,
            );
        }

        Ok(main)
    }

    pub fn rope_type(&self) -> Option<RopeType> {
        match &self.mixer {
            MixerExecutables::Attention {
                rope,
                ..
            } => Some(rope.rope_type()),
            MixerExecutables::StateSpace {
                ..
            }
            | MixerExecutables::ShortConv {
                ..
            }
            | MixerExecutables::DeltaNet {
                ..
            } => None,
        }
    }
}

pub struct LayerArguments<'a, B: Backend> {
    pub context: &'a B::Context,
    pub batch_dim: usize,
    pub token_positions: &'a Allocation<B>,
    pub token_parents: &'a Allocation<B>,
    pub token_subtrie_ranges: Option<&'a Allocation<B>>,
    pub attention_sinks: Option<&'a Allocation<B>>,
    pub rope_cosines: Option<&'a Allocation<B>>,
    pub rope_sines: Option<&'a Allocation<B>>,
    pub rope_max_sequence_length: usize,
    pub rope_dim: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub cache_layer: Option<&'a mut CacheLayer<B>>,
    #[cfg(feature = "tracing")]
    pub trace: Option<Rc<RefCell<crate::forward_pass::traces::LayerActivationTrace<B>>>>,
}
