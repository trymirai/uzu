//! Decoder executables - combines embedding, layers, normalization, and readout.

use std::rc::Rc;

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{Backend, Encoder},
    config::{DecoderConfig, DecoderLayerType, MixerConfig},
    encodable_block::{Embedding, EncodingParameters, LayerExecutables, RMSNorm, Rope, embedding::EmbeddingError},
    forward_pass::{
        model_shape::ModelShape,
        state::{ArrayId, ForwardPassState, RopeType},
    },
    parameters::ParameterTree,
};

#[derive(Debug, Error)]
pub enum DecoderError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
}

/// Full decoder executable with all layers and components.
pub struct Decoder<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Box<[LayerExecutables<B>]>,
    pub norm: RMSNorm<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B::Context>,
    ) -> Self {
        let embedding_weight_loader = root_weight_loader.subtree("embedding").expect("Failed to get embedding subtree");

        let embed = Embedding::new(
            context,
            decoder_config.vocab_size as u32,
            decoder_config.model_dim as u32,
            &decoder_config.embedding_config,
            &embedding_weight_loader,
        )
        .expect("Failed to create embedding");

        let (layers, norm) =
            Self::build_transformer_layers_and_norm(context, decoder_config, root_weight_loader, "transformer");

        Self {
            embed,
            layers,
            norm,
        }
    }

    /// Used by models whose token lookup weights and logits readout weights
    #[cfg(all(feature = "audio-runtime", metal_backend))]
    pub fn new_with_embedding_and_readout_subtrees(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B::Context>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
    ) -> Self {
        let embedding_weight_loader =
            root_weight_loader.subtree(embedding_subtree).expect("Failed to get embedding subtree");
        let readout_weight_loader = root_weight_loader.subtree(readout_subtree).expect("Failed to get readout subtree");

        let embed = Embedding::new_with_lookup_and_readout_trees(
            context,
            decoder_config.vocab_size as u32,
            decoder_config.model_dim as u32,
            &decoder_config.embedding_config,
            &embedding_weight_loader,
            &readout_weight_loader,
        )
        .expect("Failed to create embedding");

        let (layers, norm) =
            Self::build_transformer_layers_and_norm(context, decoder_config, root_weight_loader, transformer_subtree);

        Self {
            embed,
            layers,
            norm,
        }
    }

    pub(crate) fn build_transformer_layers_and_norm(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B::Context>,
        transformer_subtree: &str,
    ) -> (Box<[LayerExecutables<B>]>, RMSNorm<B>) {
        let decoder_weight_loader =
            root_weight_loader.subtree(transformer_subtree).expect("transformer subtree not found");

        let attention_data_type = Self::attention_data_type(&decoder_config);
        let norm_reference_layer =
            decoder_config.layer_configs.as_ref().map(|configs| &configs[0]).unwrap_or(&decoder_config.layer_config);
        let norm_data_type: DataType = match &norm_reference_layer.mixer_config {
            MixerConfig::Attention(attention_config) => {
                attention_config.qkv_projection_config.activation_precision().into()
            },
            MixerConfig::Mamba(mamba_config) => mamba_config.in_projection_config.activation_precision().into(),
            MixerConfig::ShortConv(short_conv_config) => {
                short_conv_config.in_projection_config.activation_precision().into()
            },
            MixerConfig::DeltaNet(config) => config.in_proj_config.activation_precision().into(),
        };

        let global_rope = if decoder_config.global_rope_config.is_some() {
            attention_data_type
                .as_ref()
                .map(|data_type| Self::create_rope_block(&context, *data_type, RopeType::Global))
        } else {
            None
        };

        let local_rope = if decoder_config.local_rope_config.is_some() {
            attention_data_type.as_ref().map(|data_type| Self::create_rope_block(&context, *data_type, RopeType::Local))
        } else {
            None
        };

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let sliding_window_sizes = model_shape.sliding_window_length_per_layer.clone();

        let layers = (0..decoder_config.num_layers)
            .map(|layer_index| {
                let layer_config = decoder_config
                    .layer_configs
                    .as_ref()
                    .map(|configs| &configs[layer_index])
                    .unwrap_or(&decoder_config.layer_config);
                let layer_type = model_shape.layer_type(layer_index);
                let rope_for_layer = match layer_type {
                    DecoderLayerType::Transformer => {
                        if let Some(_) = sliding_window_sizes[layer_index]
                            && let Some(local_rope_block) = local_rope.clone()
                        {
                            Some(local_rope_block)
                        } else {
                            Some(global_rope.clone().expect("Global rope missing for transformer layer"))
                        }
                    },
                    DecoderLayerType::StateSpace {
                        ..
                    } => None,
                    DecoderLayerType::ShortConv {
                        ..
                    } => None,
                    DecoderLayerType::DeltaNet {
                        ..
                    } => None,
                };

                let layer_loader = decoder_weight_loader.subtree(&format!("layers.{}", layer_index)).unwrap();

                LayerExecutables::new(
                    context,
                    layer_config,
                    layer_type,
                    layer_index,
                    decoder_config.model_dim,
                    decoder_config.hidden_dim,
                    decoder_config.num_heads,
                    decoder_config.head_dim,
                    decoder_config.num_groups,
                    decoder_config.attention_scale,
                    &layer_loader,
                    rope_for_layer,
                )
            })
            .collect::<Vec<_>>();

        let norm_block = RMSNorm::new(
            context,
            norm_data_type,
            decoder_config.output_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &decoder_weight_loader.subtree("output_norm").unwrap(),
            Some(ArrayId::Shortcut),
            true,
        )
        .map(RMSNorm::with_sampling_range)
        .expect("Failed to create output RMS norm kernel");

        (layers.into_boxed_slice(), norm_block)
    }

    fn create_rope_block(
        context: &B::Context,
        data_type: DataType,
        rope_type: RopeType,
    ) -> Rc<Rope<B>> {
        Rc::new(Rope::<B>::new(context, data_type, rope_type).expect("Failed to create Rope"))
    }

    fn attention_data_type(decoder_config: &DecoderConfig) -> Option<DataType> {
        (0..decoder_config.num_layers).find_map(|layer_index| {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            layer_config
                .attention_config()
                .map(|attention_config| attention_config.qkv_projection_config.activation_precision().into())
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        encoder: &mut Encoder<B>,
    ) -> Result<(), DecoderError<B>> {
        self.embed.encode_lookup(state, encoder)?;

        for layer in self.layers.iter() {
            layer.encode(state, parameters, encoder).map_err(DecoderError::BackendError)?;
        }

        if state.is_prefilling() {
            return Ok(());
        }

        self.norm.encode(state, encoder).map_err(DecoderError::BackendError)?;
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(encoder, ArrayId::Main, traces.borrow().output_norm.clone());
        }

        self.embed.encode_readout(state, encoder)?;
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(encoder, ArrayId::Logits, traces.borrow().logits.clone());
        }
        Ok(())
    }
}
