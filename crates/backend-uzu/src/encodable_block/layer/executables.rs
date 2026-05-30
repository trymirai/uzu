//! Layer executables - a single decoder layer with mixer, norms, and MLP.

use std::rc::Rc;

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{Allocation, Backend, Encoder},
    config::{
        token_mixer::AnyTokenMixerConfig, transformer::TransformerConfig, transformer_layer::TransformerLayerConfig,
    },
    encodable_block::{
        Attention, AttentionError, DeltaNetArguments, DeltaNetMixer, DeltaNetMixerError, MambaArguments, MambaMixer,
        MambaMixerError, Mlp, MlpBlockError, PerLayerEmbeddingProjection, PostLayerScalar, QkUnpack, RMSNorm,
        RMSNormError, Rope, ShortConvArguments, ShortConvMixer, ShortConvMixerError, layer::MixerExecutables,
    },
    forward_pass::{cache_layers::LayerCacheAccess, state::RopeBuffers},
    parameters::{ParameterLoaderError, ParameterTree},
};
#[cfg(feature = "tracing")]
use crate::{
    backends::common::{Kernels, kernel::TensorAddBiasKernel},
    forward_pass::traces::LayerActivationTrace,
};

#[derive(Debug, Error)]
pub enum LayerExecutablesError<B: Backend> {
    #[cfg(feature = "tracing")]
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Attention error: {0}")]
    AttentionError(#[from] AttentionError<B>),
    #[error("Mamba mixer error: {0}")]
    MambaMixerError(#[from] MambaMixerError<B>),
    #[error("ShortConv mixer error: {0}")]
    ShortConvMixerError(#[from] ShortConvMixerError<B>),
    #[error("DeltaNet mixer error: {0}")]
    DeltaNetMixerError(#[from] DeltaNetMixerError<B>),
    #[error("MLP error: {0}")]
    MlpBlockError(#[from] MlpBlockError<B>),
    #[error("RMSNorm error: {0}")]
    RMSNormError(#[from] RMSNormError<B>),
    #[error("layer {layer_index} sets post_layer_scalar but has no post_mlp_norm")]
    PostLayerScalarWithoutPostMlpNorm {
        layer_index: usize,
    },
    #[error("Unsupported post_layer_scalar data type: {data_type:?}")]
    UnsupportedPostLayerScalarDataType {
        data_type: DataType,
    },
    #[error("decoder layers require pre_mixer_norm_config")]
    MissingPreMixerNormConfig,
}

/// A single decoder layer with all its components.
pub struct LayerExecutables<B: Backend> {
    pub layer_index: usize,
    #[cfg(feature = "tracing")]
    pub tensor_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    pub pre_mixer_norm: RMSNorm<B>,
    mixer: MixerExecutables<B>,
    pub post_mixer_norm: Option<RMSNorm<B>>,
    pub pre_mlp_norm: RMSNorm<B>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<RMSNorm<B>>,
    pub ple_projection: Option<PerLayerEmbeddingProjection<B>>,
    #[cfg(feature = "tracing")]
    model_dim: usize,
}

impl<B: Backend> LayerExecutables<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        transformer_config: &TransformerConfig,
        layer_config: &TransformerLayerConfig,
        layer_index: usize,
        parameter_tree: &ParameterTree<B>,
        rope: &Rc<Rope<B>>,
        qk_unpack: &Rc<QkUnpack<B>>,
        data_type: DataType,
    ) -> Result<Self, LayerExecutablesError<B>> {
        let post_layer_scalar = if layer_config.has_post_layer_scalar {
            if layer_config.post_mlp_norm_config.is_none() {
                return Err(LayerExecutablesError::PostLayerScalarWithoutPostMlpNorm {
                    layer_index,
                });
            }
            let leaf = parameter_tree.leaf("post_layer_scalar")?;
            let scalar = match data_type {
                DataType::F32 => leaf.validate(&[1], data_type)?.read_slice::<f32>()?[0],
                DataType::F16 => leaf.validate(&[1], data_type)?.read_slice::<half::f16>()?[0].to_f32(),
                DataType::BF16 => leaf.validate(&[1], data_type)?.read_slice::<half::bf16>()?[0].to_f32(),
                data_type => {
                    return Err(LayerExecutablesError::UnsupportedPostLayerScalarDataType {
                        data_type,
                    });
                },
            };
            Some(scalar)
        } else {
            None
        };

        // When a PLE projection is present it owns the post-layer scalar (applied
        // to the combined residual), so the norms must not also apply it.
        let (residual_sum_scalar, output_scalar) = match (post_layer_scalar, layer_config.ple_config.is_none()) {
            (Some(scalar), true) => (PostLayerScalar::ScaleResidualSum(scalar), PostLayerScalar::ScaleOutput(scalar)),
            _ => (PostLayerScalar::None, PostLayerScalar::None),
        };

        #[cfg(feature = "tracing")]
        let tensor_add = TensorAddBiasKernel::new(context, data_type, data_type, false)
            .map_err(LayerExecutablesError::BackendError)?;

        let mixer_subtree = parameter_tree.subtree("mixer")?;

        let (mixer, mixer_hadamard_factors) = match &layer_config.mixer_config {
            AnyTokenMixerConfig::AttentionConfig(attention_config) => {
                let (attention, input_hadamard_factors) = Attention::new(
                    context,
                    transformer_config.model_dim,
                    data_type,
                    attention_config,
                    &mixer_subtree,
                    rope.clone(),
                    qk_unpack.clone(),
                    attention_config.gate_projection_config.is_none(),
                )?;

                (
                    MixerExecutables::Attention {
                        attention,
                    },
                    input_hadamard_factors,
                )
            },
            AnyTokenMixerConfig::Mamba2Config(mamba_config) => {
                let (mixer, input_hadamard_factors) =
                    MambaMixer::new(context, mamba_config, transformer_config.model_dim, &mixer_subtree, data_type)?;
                (
                    MixerExecutables::StateSpace {
                        mixer,
                    },
                    input_hadamard_factors,
                )
            },
            AnyTokenMixerConfig::ShortConvConfig(short_conv_config) => {
                let (mixer, input_hadamard_factors) = ShortConvMixer::new(
                    context,
                    short_conv_config,
                    transformer_config.model_dim,
                    &mixer_subtree,
                    data_type,
                )?;
                (
                    MixerExecutables::ShortConv {
                        mixer,
                    },
                    input_hadamard_factors,
                )
            },
            AnyTokenMixerConfig::DeltaNetConfig(delta_net_config) => {
                let (mixer, input_hadamard_factors) = DeltaNetMixer::new(
                    context,
                    delta_net_config,
                    transformer_config.model_dim,
                    &mixer_subtree,
                    data_type,
                )?;
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
            data_type,
            transformer_config.model_dim,
            layer_config.pre_mixer_norm_config.clone().ok_or(LayerExecutablesError::MissingPreMixerNormConfig)?,
            &parameter_tree.subtree("pre_mixer_norm")?,
            mixer_hadamard_factors,
            true,
            layer_index > 0,
            PostLayerScalar::None,
        )?;

        let post_mixer_norm = if let Some(norm_config) = &layer_config.post_mixer_norm_config {
            Some(RMSNorm::new(
                context,
                data_type,
                transformer_config.model_dim,
                norm_config.clone(),
                &parameter_tree.subtree("post_mixer_norm")?,
                None,
                false,
                false,
                PostLayerScalar::None,
            )?)
        } else {
            None
        };

        let (mlp, mlp_input_hadamard_factors) = <dyn Mlp<B>>::new(
            &layer_config.mlp_config,
            transformer_config.model_dim,
            layer_config.hidden_dim.unwrap_or(transformer_config.hidden_dim),
            context,
            &parameter_tree.subtree("mlp")?,
            data_type,
        )?;

        let pre_mlp_norm = RMSNorm::new(
            context,
            data_type,
            transformer_config.model_dim,
            layer_config.pre_mlp_norm_config.clone(),
            &parameter_tree.subtree("pre_mlp_norm")?,
            mlp_input_hadamard_factors,
            true,
            true,
            residual_sum_scalar,
        )?;

        let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
            Some(RMSNorm::new(
                context,
                data_type,
                transformer_config.model_dim,
                norm_config.clone(),
                &parameter_tree.subtree("post_mlp_norm")?,
                None,
                false,
                false,
                output_scalar,
            )?)
        } else {
            None
        };

        let ple_projection = layer_config.ple_config.as_ref().map(|ple_config| {
            let ple_loader = parameter_tree.subtree("ple").expect("Failed to get ple subtree");
            PerLayerEmbeddingProjection::new(
                context,
                ple_config,
                transformer_config.model_dim,
                transformer_config.layer_configs.len(),
                post_layer_scalar.unwrap_or(1.0),
                data_type,
                &ple_loader,
            )
            .expect("Failed to create per-layer embedding projection")
        });

        Ok(Self {
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
        })
    }

    pub fn encode(
        &self,
        args: LayerArguments<B>,
        input: Allocation<B>,
        shortcut: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let LayerArguments {
            batch_dim,
            token_positions,
            token_parents,
            token_subtrie_ranges,
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

        let mut hidden = self.pre_mixer_norm.encode(&input, 0, batch_dim, Some(shortcut), encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(shortcut, .., layer_traces.inputs.allocation_mut(), ..);
            encoder.encode_copy(&hidden, .., layer_traces.pre_attention_norm.allocation_mut(), ..);
        }

        hidden = match &self.mixer {
            MixerExecutables::Attention {
                attention,
            } => attention.encode(
                token_positions,
                token_subtrie_ranges,
                rope_buffers,
                cache_access,
                hidden,
                batch_dim,
                encoder,
            )?,
            MixerExecutables::StateSpace {
                mixer,
            } => {
                let Some(LayerCacheAccess::Owned {
                    entry,
                }) = cache_access
                else {
                    panic!("State-space layer requires writable cache state");
                };
                let layer = entry.as_state_space_mut().expect("State-space mixer expects SSM cache layer");
                mixer.encode(
                    MambaArguments {
                        active_row_count: batch_dim,
                        layer,
                    },
                    hidden,
                    encoder,
                )?
            },
            MixerExecutables::ShortConv {
                mixer,
            } => {
                let Some(LayerCacheAccess::Owned {
                    entry,
                }) = cache_access
                else {
                    panic!("ShortConv layer requires writable cache state");
                };
                let layer = entry.as_short_conv_mut().expect("ShortConv mixer expects ShortConv cache layer");
                mixer.encode(
                    ShortConvArguments {
                        active_row_count: batch_dim,
                        sampling_start,
                        sampling_length,
                        token_parents,
                        layer,
                    },
                    hidden,
                    encoder,
                )?
            },
            MixerExecutables::DeltaNet {
                mixer,
            } => {
                let Some(LayerCacheAccess::Owned {
                    entry,
                }) = cache_access
                else {
                    panic!("DeltaNet layer requires writable cache state");
                };
                let layer = entry.as_delta_net_mut().expect("DeltaNet mixer expects DeltaNet cache layer");
                mixer.encode(
                    DeltaNetArguments {
                        active_row_count: batch_dim,
                        layer,
                    },
                    hidden,
                    encoder,
                )?
            },
        };
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&hidden, .., layer_traces.attention.allocation_mut(), ..);
        }

        if let Some(post_mixer_norm) = &self.post_mixer_norm {
            hidden = post_mixer_norm.encode(&hidden, 0, batch_dim, None, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(layer_traces) = layer_traces.as_deref_mut() {
                encoder.encode_copy(&hidden, .., layer_traces.post_attention_norm.allocation_mut(), ..);
            }
        }

        hidden = self.pre_mlp_norm.encode(&hidden, 0, batch_dim, Some(shortcut), encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(shortcut, .., layer_traces.mlp_inputs.allocation_mut(), ..);
            encoder.encode_copy(&hidden, .., layer_traces.pre_mlp_norm.allocation_mut(), ..);
        }

        hidden = self.mlp.encode(hidden, batch_dim, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(layer_traces) = layer_traces.as_deref_mut() {
            encoder.encode_copy(&hidden, .., layer_traces.mlp.allocation_mut(), ..);
        }

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            hidden = post_mlp_norm.encode(&hidden, 0, batch_dim, None, encoder)?;
            #[cfg(feature = "tracing")]
            if let Some(layer_traces) = layer_traces.as_deref_mut() {
                encoder.encode_copy(&hidden, .., layer_traces.post_mlp_norm.allocation_mut(), ..);
            }
        }

        if let Some(ple_projection) = &self.ple_projection {
            let per_layer_inputs = per_layer_inputs.expect("per-layer inputs required for PLE layer");
            ple_projection.encode(self.layer_index, per_layer_inputs, shortcut, &hidden, batch_dim, encoder)?;
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
    pub rope_buffers: Option<&'a RopeBuffers<B>>,
    pub per_layer_inputs: Option<&'a Allocation<B>>,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub cache_access: Option<LayerCacheAccess<'a, B>>,
    #[cfg(feature = "tracing")]
    pub trace: Option<&'a mut LayerActivationTrace<B>>,
}
