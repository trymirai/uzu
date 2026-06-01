use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::{transformer::TransformerConfig, transformer_layer::TransformerLayerConfig},
    data_type::DataType,
    encodable_block::{
        mixer::{Mixer, MixerNewError, MixerState, MixerTokenTopology, attention::rope::PrecalculatedRoPE},
        mlp::{Mlp, MlpBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
        per_layer_embedding::PerLayerEmbeddingProjection,
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

#[derive(Debug, Error)]
pub enum TransformerLayerError<B: Backend> {
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Mixer error: {0}")]
    Mixer(#[from] MixerNewError<B>),
    #[error("MLP error: {0}")]
    MlpBlock(#[from] MlpBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Layer {layer_index} sets post_layer_scalar but has no post_mlp_norm")]
    PostLayerScalarWithoutPostMlpNorm {
        layer_index: usize,
    },
    #[error("Transformer layers except the first one if it doesn't have rht in mixer require pre_mixer_norm_config")]
    MissingPreMixerNormConfig,
}

// TODO: saner shortcut

pub struct TransformerLayer<B: Backend> {
    pub layer_index: usize,
    pub pre_mixer_norm: Option<Normalization<B>>,
    pub kv_source_layer_index: Option<usize>,
    pub mixer: Box<dyn Mixer<B>>,
    pub post_mixer_norm: Option<Normalization<B>>,
    pub pre_mlp_norm: Normalization<B>,
    pub mlp: Box<dyn Mlp<B>>,
    pub post_mlp_norm: Option<Normalization<B>>,
    pub ple_projection: Option<PerLayerEmbeddingProjection<B>>,
}

impl<B: Backend> TransformerLayer<B> {
    pub fn new(
        context: &B::Context,
        transformer_config: &TransformerConfig,
        layer_config: &TransformerLayerConfig,
        layer_index: usize,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, TransformerLayerError<B>> {
        let post_layer_scalar = if layer_config.has_post_layer_scalar {
            if layer_config.post_mlp_norm_config.is_none() {
                return Err(TransformerLayerError::PostLayerScalarWithoutPostMlpNorm {
                    layer_index,
                });
            }
            let leaf = parameter_tree.leaf("post_layer_scalar")?;
            let scalar = match data_type {
                DataType::F32 => leaf.validate(&[1], data_type)?.read_slice::<f32>()?[0],
                DataType::F16 => leaf.validate(&[1], data_type)?.read_slice::<half::f16>()?[0].to_f32(),
                DataType::BF16 => leaf.validate(&[1], data_type)?.read_slice::<half::bf16>()?[0].to_f32(),
                _ => unreachable!(),
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

        let (mixer, mixer_hadamard_factors) = <dyn Mixer<B>>::new(
            transformer_config.model_dim,
            data_type,
            layer_config.rope_config.as_ref(),
            &layer_config.mixer_config,
            &parameter_tree.subtree("mixer")?,
            context,
        )?;

        let pre_mixer_norm = if let Some(pre_mixer_norm_config) = &layer_config.pre_mixer_norm_config {
            Some(Normalization::new(
                transformer_config.model_dim,
                mixer_hadamard_factors,
                true,
                layer_index > 0,
                PostLayerScalar::None,
                data_type,
                pre_mixer_norm_config,
                &parameter_tree.subtree("pre_mixer_norm")?,
                context,
            )?)
        } else {
            if layer_index != 0 || mixer_hadamard_factors.is_some() {
                return Err(TransformerLayerError::MissingPreMixerNormConfig);
            }
            None
        };

        let post_mixer_norm = if let Some(norm_config) = &layer_config.post_mixer_norm_config {
            Some(Normalization::new(
                transformer_config.model_dim,
                None,
                false,
                false,
                PostLayerScalar::None,
                data_type,
                norm_config,
                &parameter_tree.subtree("post_mixer_norm")?,
                context,
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

        let pre_mlp_norm = Normalization::new(
            transformer_config.model_dim,
            mlp_input_hadamard_factors,
            true,
            true,
            residual_sum_scalar,
            data_type,
            &layer_config.pre_mlp_norm_config,
            &parameter_tree.subtree("pre_mlp_norm")?,
            context,
        )?;

        let post_mlp_norm = if let Some(norm_config) = &layer_config.post_mlp_norm_config {
            Some(Normalization::new(
                transformer_config.model_dim,
                None,
                false,
                false,
                output_scalar,
                data_type,
                norm_config,
                &parameter_tree.subtree("post_mlp_norm")?,
                context,
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
            pre_mixer_norm,
            kv_source_layer_index: layer_config.kv_source_layer_index,
            mixer,
            post_mixer_norm,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
            ple_projection,
        })
    }

    pub fn encode(
        &self,
        input: Allocation<B>,
        shortcut: &mut Allocation<B>,
        per_layer_inputs: Option<&Allocation<B>>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        token_topology: &MixerTokenTopology<B>,
        batch_dim: usize,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let hidden = if let Some(pre_mixer_norm) = &self.pre_mixer_norm {
            pre_mixer_norm.encode(&input, 0, batch_dim, Some(shortcut), encoder)?
        } else {
            assert!(self.layer_index == 0);
            encoder.encode_copy(&input, .., shortcut, ..);
            input
        };

        // TODO: In prefill outside of sampling suffix in last layer part of mixer (ie out projection) and everything after is dead code
        let mut hidden = self.mixer.encode(hidden, precalculated_rope, token_topology, batch_dim, state, encoder)?;

        if let Some(post_mixer_norm) = &self.post_mixer_norm {
            hidden = post_mixer_norm.encode(&hidden, 0, batch_dim, None, encoder)?;
        }

        hidden = self.pre_mlp_norm.encode(&hidden, 0, batch_dim, Some(shortcut), encoder)?;

        hidden = self.mlp.encode(hidden, batch_dim, encoder)?;

        if let Some(post_mlp_norm) = &self.post_mlp_norm {
            hidden = post_mlp_norm.encode(&hidden, 0, batch_dim, None, encoder)?;
        }

        if let Some(ple_projection) = &self.ple_projection {
            let per_layer_inputs = per_layer_inputs.expect("per-layer inputs required for PLE layer");
            ple_projection.encode(self.layer_index, per_layer_inputs, shortcut, &hidden, batch_dim, encoder)?;
            encoder.encode_fill(&mut hidden, 0);
        }

        Ok(hidden)
    }
}
