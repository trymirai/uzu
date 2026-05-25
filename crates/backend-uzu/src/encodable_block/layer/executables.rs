//! Layer executables - a single decoder layer with mixer, norms, and MLP.

use std::rc::Rc;

use half::{bf16, f16};
use thiserror::Error;

use super::MixerExecutables;
#[cfg(feature = "tracing")]
use crate::backends::common::{Kernels, kernel::TensorAddBiasKernel};
#[cfg(feature = "tracing")]
use crate::forward_pass::traces::LayerActivationTrace;
use crate::{
    DataType,
    backends::common::{Allocation, AsBufferRangeRef, Backend, Encoder},
    config::{MixerConfig, TransformerConfig, TransformerLayerConfig},
    encodable_block::{
        Attention, AttentionArguments, DeltaNetArguments, DeltaNetMixer, Linear, MambaArguments, MambaMixer, Mlp,
        PerLayerEmbeddingProjection, PostLayerScalar, QKVNorm, QkUnpack, RMSNorm, Rope, ShortConvArguments,
        ShortConvMixer,
    },
    forward_pass::{cache_layers::LayerCacheAccess, state::RopeBuffers},
    parameters::ParameterTree,
};

/// A single decoder layer with all its components.
pub struct LayerExecutables<B: Backend> {
    pub layer_index: usize,
    #[cfg(feature = "tracing")]
    pub tensor_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    pub pre_mixer_norm: RMSNorm<B>,
    pub(crate) mixer: MixerExecutables<B>,
    pub post_mixer_norm: Option<RMSNorm<B>>,
    pub pre_mlp_norm: RMSNorm<B>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<RMSNorm<B>>,
    pub ple_projection: Option<PerLayerEmbeddingProjection<B>>,
    #[cfg(feature = "tracing")]
    model_dim: usize,
}

#[derive(Debug, Error)]
pub enum LayerEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Missing cache for {mixer} layer {layer_index}")]
    MissingCacheState {
        layer_index: usize,
        mixer: &'static str,
    },
    #[error("Invalid cache for {mixer} layer {layer_index}")]
    InvalidCacheLayer {
        layer_index: usize,
        mixer: &'static str,
    },
    #[error("Missing per-layer inputs for layer {layer_index}")]
    MissingPerLayerInputs {
        layer_index: usize,
    },
}

impl<B: Backend> LayerExecutables<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        transformer_config: &TransformerConfig,
        layer_config: &TransformerLayerConfig,
        layer_index: usize,
        decoder_layer_loader: &ParameterTree<B::Context>,
        rope: &Rc<Rope<B>>,
        qk_unpack: &Rc<QkUnpack<B>>,
    ) -> Self {
        let intermediate_data_type: DataType = layer_config.mixer_config.activation_precision().into();

        let post_layer_scalar = if layer_config.has_post_layer_scalar {
            assert!(
                layer_config.post_mlp_norm_config.is_some(),
                "layer {layer_index} sets post_layer_scalar but has no post_mlp_norm"
            );
            let leaf = decoder_layer_loader.leaf("post_layer_scalar").expect("Failed to read post_layer_scalar weight");
            Some(match leaf.data_type() {
                DataType::BF16 => {
                    leaf.read_slice::<bf16>().expect("Failed to read post_layer_scalar weight")[0].to_f32()
                },
                DataType::F16 => leaf.read_slice::<f16>().expect("Failed to read post_layer_scalar weight")[0].to_f32(),
                DataType::F32 => leaf.read_slice::<f32>().expect("Failed to read post_layer_scalar weight")[0],
                other => panic!("post_layer_scalar must be a float dtype, got {other:?}"),
            })
        } else {
            None
        };

        let (residual_sum_scalar, output_scalar) = match (post_layer_scalar, layer_config.ple_config.is_none()) {
            (Some(scalar), true) => (PostLayerScalar::ScaleResidualSum(scalar), PostLayerScalar::ScaleOutput(scalar)),
            _ => (PostLayerScalar::None, PostLayerScalar::None),
        };

        #[cfg(feature = "tracing")]
        let tensor_add = TensorAddBiasKernel::new(context, intermediate_data_type, false)
            .expect("Failed to create TensorAddBiasKernel kernel"); // TODO: this function return Result

        let (mixer, mixer_hadamard_factors) = match &layer_config.mixer_config {
            MixerConfig::Attention(attention_config) => {
                let q_dim = attention_config.num_heads * attention_config.head_dim;
                let kv_dim = attention_config.num_groups * attention_config.head_dim;

                let (qkv_projection, input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                    &attention_config.qkv_projection_config,
                    transformer_config.model_dim,
                    [q_dim, kv_dim, kv_dim],
                    context,
                    &decoder_layer_loader.subtree("mixer.qkv_projection").unwrap(),
                )
                .expect("Failed to create qkv projection");

                let gate_projection = attention_config.gate_projection_config.as_ref().map(|gate_config| {
                    let gate_tree = decoder_layer_loader.subtree("mixer.gate_projection").unwrap();
                    match (input_hadamard_factors.is_some(), gate_config) {
                        (
                            true,
                            crate::config::LinearConfig::RHTLinearWrapper {
                                inner_config,
                                ..
                            },
                        ) => {
                            let output_factors = gate_tree
                                .leaf("output_factors")
                                .expect("Failed to get gate projection output_factors")
                                .read_allocation()
                                .expect("Failed to read gate projection output_factors");
                            let inner_tree = gate_tree
                                .subtree("inner_linear")
                                .expect("Failed to get gate projection inner_linear subtree");
                            <dyn Linear<B>>::new_with_output_hadamard(
                                context,
                                inner_config,
                                &inner_tree,
                                output_factors,
                                transformer_config.model_dim,
                                q_dim,
                            )
                        },
                        (
                            false,
                            crate::config::LinearConfig::RHTLinearWrapper {
                                ..
                            },
                        )
                        | (true, _) => {
                            panic!("attention qkv/gate projections must share input hadamard")
                        },
                        (false, _) => <dyn Linear<B>>::new(
                            gate_config,
                            transformer_config.model_dim,
                            [q_dim],
                            context,
                            &gate_tree,
                        ),
                    }
                    .expect("Failed to create gate projection")
                });

                let value_norm_config = attention_config.value_norm_config();
                let qkv_norm = if attention_config.query_norm_config.is_some()
                    || attention_config.key_norm_config.is_some()
                    || value_norm_config.is_some()
                {
                    match QKVNorm::new(
                        context,
                        intermediate_data_type,
                        attention_config.query_norm_config.clone(),
                        attention_config.key_norm_config.clone(),
                        value_norm_config,
                        &decoder_layer_loader.subtree("mixer").unwrap(),
                        attention_config.num_heads,
                        attention_config.num_groups,
                        attention_config.head_dim,
                    ) {
                        Ok(qkv_norm) => Some(qkv_norm),
                        Err(error) => {
                            panic!("Failed to create QKV norm kernel for layer {}: {:?}", layer_index, error)
                        },
                    }
                } else {
                    None
                };

                let out_projection = <dyn Linear<B>>::new(
                    &attention_config.out_projection_config,
                    q_dim,
                    [transformer_config.model_dim],
                    context,
                    &decoder_layer_loader.subtree("mixer.out_projection").unwrap(),
                )
                .expect("Failed to create out projection");

                let attention =
                    Attention::new(context, intermediate_data_type, attention_config, gate_projection.is_some())
                        .expect("Failed to create AttentionWrapper kernel");

                (
                    MixerExecutables::Attention {
                        qkv_projection,
                        gate_projection,
                        qkv_norm,
                        rope: rope.clone(),
                        qk_unpack: qk_unpack.clone(),
                        attention,
                        out_projection,
                        num_heads: attention_config.num_heads,
                        num_groups: attention_config.num_groups,
                        head_dim: attention_config.head_dim,
                    },
                    input_hadamard_factors,
                )
            },
            MixerConfig::Mamba(mamba_config) => {
                let (mixer, input_hadamard_factors) =
                    MambaMixer::new(context, mamba_config.clone(), transformer_config.model_dim, decoder_layer_loader)
                        .expect("Failed to create Mamba mixer");
                (
                    MixerExecutables::StateSpace {
                        mixer,
                    },
                    input_hadamard_factors,
                )
            },
            MixerConfig::ShortConv(short_conv_config) => {
                let (mixer, input_hadamard_factors) = ShortConvMixer::new(
                    context,
                    short_conv_config.clone(),
                    transformer_config.model_dim,
                    decoder_layer_loader,
                )
                .expect("Failed to create ShortConv mixer");
                (
                    MixerExecutables::ShortConv {
                        mixer,
                    },
                    input_hadamard_factors,
                )
            },
            MixerConfig::DeltaNet(delta_net_config) => {
                let (mixer, input_hadamard_factors) = DeltaNetMixer::new(
                    context,
                    delta_net_config.clone(),
                    transformer_config.model_dim,
                    decoder_layer_loader,
                )
                .expect("Failed to create DeltaNet mixer");
                (
                    MixerExecutables::DeltaNet {
                        mixer,
                    },
                    input_hadamard_factors,
                )
            },
        };

        let pre_mixer_norm = RMSNorm::new(
            context,
            intermediate_data_type,
            layer_config.pre_mixer_norm_config.clone().expect("decoder layers require pre_mixer_norm_config"),
            &decoder_layer_loader.subtree("pre_mixer_norm").unwrap(),
            mixer_hadamard_factors,
            true,
            layer_index > 0,
            PostLayerScalar::None,
        )
        .expect("Failed to create RMS norm kernel");

        let post_mixer_norm = if let Some(norm_config) = &layer_config.post_mixer_norm_config {
            Some(
                RMSNorm::new(
                    context,
                    intermediate_data_type,
                    norm_config.clone(),
                    &decoder_layer_loader.subtree("post_mixer_norm").unwrap(),
                    None,
                    false,
                    false,
                    PostLayerScalar::None,
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
        };

        let (mlp, mlp_input_hadamard_factors) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            transformer_config.model_dim,
            layer_config.hidden_dim.unwrap_or(transformer_config.hidden_dim),
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
            residual_sum_scalar,
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
                    output_scalar,
                )
                .expect("Failed to create RMS norm kernel"),
            )
        } else {
            None
        };

        let ple_projection = layer_config.ple_config.as_ref().map(|ple_config| {
            PerLayerEmbeddingProjection::new(
                context,
                ple_config,
                transformer_config.model_dim,
                transformer_config.layer_configs.len(),
                post_layer_scalar.unwrap_or(1.0),
                &decoder_layer_loader.subtree("ple").unwrap(),
            )
            .expect("Failed to create per-layer embedding projection")
        });

        Self {
            layer_index,
            #[cfg(feature = "tracing")]
            tensor_add,
            pre_mixer_norm,
            mixer,
            post_mixer_norm,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
            ple_projection,
            #[cfg(feature = "tracing")]
            model_dim: transformer_config.model_dim,
        }
    }

    pub fn encode(
        &self,
        args: LayerArguments<B>,
        input: Allocation<B>,
        shortcut: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, LayerEncodeError<B>> {
        let LayerArguments {
            batch_dim,
            token_positions,
            token_parents,
            token_subtrie_ranges,
            attention_sinks,
            rope_buffers,
            per_layer_inputs,
            sampling_start,
            sampling_length,
            cache_access,
            #[cfg(feature = "tracing")]
            trace,
        } = args;
        #[cfg(feature = "tracing")]
        let mut layer_traces = trace;

        let mut hidden = self
            .pre_mixer_norm
            .encode(&input, 0, batch_dim, Some(shortcut), encoder)
            .map_err(LayerEncodeError::BackendError)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(shortcut, .., layer_traces.inputs.allocation_mut(), ..);
            encoder.encode_copy(&hidden, .., layer_traces.pre_attention_norm.allocation_mut(), ..);
        }

        hidden = match &self.mixer {
            MixerExecutables::Attention {
                qkv_projection,
                gate_projection,
                qkv_norm,
                rope,
                qk_unpack,
                attention,
                out_projection,
                num_heads,
                num_groups,
                head_dim,
            } => {
                let gate_input = if gate_projection.is_some() {
                    let hidden_len = hidden.as_buffer_range_ref().range().len();
                    let mut gate_input =
                        encoder.allocate_scratch(hidden_len).map_err(LayerEncodeError::BackendError)?;
                    encoder.encode_copy(&hidden, .., &mut gate_input, ..);
                    Some(gate_input)
                } else {
                    None
                };
                let mut qkv =
                    qkv_projection.encode(hidden, batch_dim, encoder).map_err(LayerEncodeError::BackendError)?;
                let gate = match (gate_projection, gate_input) {
                    (Some(gate_proj), Some(gate_input)) => {
                        Some(gate_proj.encode(gate_input, batch_dim, encoder).map_err(LayerEncodeError::BackendError)?)
                    },
                    _ => None,
                };
                if let Some(norm) = qkv_norm {
                    norm.encode(&mut qkv, batch_dim, encoder).map_err(LayerEncodeError::BackendError)?;
                }
                let (queries, rotated_keys) = match rope_buffers {
                    Some(rope_buffers) => rope
                        .encode(
                            &qkv,
                            token_positions,
                            &rope_buffers.cosines,
                            &rope_buffers.sines,
                            batch_dim,
                            *num_heads,
                            *num_groups,
                            *head_dim,
                            rope_buffers.max_sequence_length(),
                            rope_buffers.dim(),
                            encoder,
                        )
                        .map_err(LayerEncodeError::BackendError)?,
                    None => qk_unpack
                        .encode(&qkv, batch_dim, *num_heads, *num_groups, *head_dim, encoder)
                        .map_err(LayerEncodeError::BackendError)?,
                };
                let attention_output = attention
                    .encode(
                        AttentionArguments {
                            token_subtrie_ranges,
                            attention_sinks,
                            cache_access,
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
                    )
                    .map_err(LayerEncodeError::BackendError)?;
                out_projection.encode(attention_output, batch_dim, encoder).map_err(LayerEncodeError::BackendError)?
            },
            MixerExecutables::StateSpace {
                mixer,
            } => {
                let Some(LayerCacheAccess::Owned {
                    entry,
                }) = cache_access
                else {
                    return Err(LayerEncodeError::MissingCacheState {
                        layer_index: self.layer_index,
                        mixer: "state-space",
                    });
                };
                let layer = entry.as_state_space_mut().ok_or(LayerEncodeError::InvalidCacheLayer {
                    layer_index: self.layer_index,
                    mixer: "state-space",
                })?;
                mixer
                    .encode(
                        MambaArguments {
                            active_row_count: batch_dim,
                            layer,
                        },
                        hidden,
                        encoder,
                    )
                    .map_err(LayerEncodeError::BackendError)?
            },
            MixerExecutables::ShortConv {
                mixer,
            } => {
                let Some(LayerCacheAccess::Owned {
                    entry,
                }) = cache_access
                else {
                    return Err(LayerEncodeError::MissingCacheState {
                        layer_index: self.layer_index,
                        mixer: "short-conv",
                    });
                };
                let layer = entry.as_short_conv_mut().ok_or(LayerEncodeError::InvalidCacheLayer {
                    layer_index: self.layer_index,
                    mixer: "short-conv",
                })?;
                mixer
                    .encode(
                        ShortConvArguments {
                            active_row_count: batch_dim,
                            sampling_start,
                            sampling_length,
                            token_parents,
                            layer,
                        },
                        hidden,
                        encoder,
                    )
                    .map_err(LayerEncodeError::BackendError)?
            },
            MixerExecutables::DeltaNet {
                mixer,
            } => {
                let Some(LayerCacheAccess::Owned {
                    entry,
                }) = cache_access
                else {
                    return Err(LayerEncodeError::MissingCacheState {
                        layer_index: self.layer_index,
                        mixer: "delta-net",
                    });
                };
                let layer = entry.as_delta_net_mut().ok_or(LayerEncodeError::InvalidCacheLayer {
                    layer_index: self.layer_index,
                    mixer: "delta-net",
                })?;
                mixer
                    .encode(
                        DeltaNetArguments {
                            active_row_count: batch_dim,
                            layer,
                        },
                        hidden,
                        encoder,
                    )
                    .map_err(LayerEncodeError::BackendError)?
            },
        };
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&hidden, .., layer_traces.attention.allocation_mut(), ..);
        }

        if let Some(post_mixer_norm) = &self.post_mixer_norm {
            hidden =
                post_mixer_norm.encode(&hidden, 0, batch_dim, None, encoder).map_err(LayerEncodeError::BackendError)?;
            #[cfg(feature = "tracing")]
            if let Some(layer_traces) = layer_traces.as_deref_mut() {
                encoder.encode_copy(&hidden, .., layer_traces.post_attention_norm.allocation_mut(), ..);
            }
        }

        hidden = self
            .pre_mlp_norm
            .encode(&hidden, 0, batch_dim, Some(shortcut), encoder)
            .map_err(LayerEncodeError::BackendError)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(shortcut, .., layer_traces.mlp_inputs.allocation_mut(), ..);
            encoder.encode_copy(&hidden, .., layer_traces.pre_mlp_norm.allocation_mut(), ..);
        }

        hidden = self.mlp.encode(hidden, batch_dim, encoder).map_err(LayerEncodeError::BackendError)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&hidden, .., layer_traces.mlp.allocation_mut(), ..);
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            hidden =
                post_mlp_norm.encode(&hidden, 0, batch_dim, None, encoder).map_err(LayerEncodeError::BackendError)?;
            #[cfg(feature = "tracing")]
            if let Some(layer_traces) = layer_traces.as_deref_mut() {
                encoder.encode_copy(&hidden, .., layer_traces.post_mlp_norm.allocation_mut(), ..);
            }
        }

        if let Some(ple_projection) = &self.ple_projection {
            let Some(per_layer_inputs) = per_layer_inputs else {
                return Err(LayerEncodeError::MissingPerLayerInputs {
                    layer_index: self.layer_index,
                });
            };
            ple_projection
                .encode(self.layer_index, per_layer_inputs, shortcut, &hidden, batch_dim, encoder)
                .map_err(LayerEncodeError::BackendError)?;
            encoder.encode_fill(&mut hidden, 0);
        }

        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            let size = (batch_dim * self.model_dim) as u32;
            self.tensor_add.encode(
                Some(&hidden),
                &*shortcut,
                layer_traces.outputs.allocation_mut(),
                size,
                size,
                encoder,
            );
        }

        Ok(hidden)
    }
}

pub struct LayerArguments<'a, B: Backend> {
    pub batch_dim: usize,
    pub token_positions: &'a Allocation<B>,
    pub token_parents: &'a Allocation<B>,
    pub token_subtrie_ranges: Option<&'a Allocation<B>>,
    pub attention_sinks: Option<&'a Allocation<B>>,
    pub rope_buffers: Option<&'a RopeBuffers<B>>,
    pub per_layer_inputs: Option<&'a Allocation<B>>,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub cache_access: Option<LayerCacheAccess<'a, B>>,
    #[cfg(feature = "tracing")]
    pub trace: Option<&'a mut LayerActivationTrace<B>>,
}
