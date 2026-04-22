//! Decoder executables - combines embedding, layers, normalization, and readout.

use std::rc::Rc;

use thiserror::Error;

#[cfg(feature = "tracing")]
use crate::forward_pass::traces::ActivationTrace;
use crate::{
    DataType,
    backends::common::{Allocation, Backend, Encoder},
    config::{DecoderConfig, DecoderLayerType, MixerConfig},
    encodable_block::{
        Embedding, EncodingParameters, LayerArguments, LayerExecutables, RMSNorm, Rope, embedding::EmbeddingError,
    },
    forward_pass::{
        cache_layers::CacheLayers,
        model_shape::ModelShape,
        state::{RopeType, SharedBuffers},
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

pub struct DecoderArguments<'a, B: Backend> {
    pub activation_data_type: DataType,
    pub token_ids: &'a Allocation<B>,
    pub token_positions: &'a Allocation<B>,
    pub token_parents: &'a Allocation<B>,
    pub token_subtrie_ranges: Option<&'a Allocation<B>>,
    pub shared_buffers: &'a SharedBuffers<B>,
    pub cache_layers: Option<&'a mut CacheLayers<B>>,
    pub batch_dim: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub rope_max_sequence_length: usize,
    pub rope_dim: usize,
    #[cfg(feature = "tracing")]
    pub trace: Option<&'a mut ActivationTrace<B>>,
}

pub enum DecoderDecodeInput<B: Backend> {
    TokenIds,
    Embeddings(Allocation<B>),
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
    #[cfg(metal_backend)]
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
            &decoder_weight_loader.subtree("output_norm").unwrap(),
            None,
            true,
            true,
        )
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

    fn run_layers(
        &self,
        args: DecoderArguments<B>,
        mut main: Allocation<B>,
        parameters: &EncodingParameters,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), DecoderError<B>> {
        let DecoderArguments {
            activation_data_type: _,
            token_ids: _,
            token_positions,
            token_parents,
            token_subtrie_ranges,
            shared_buffers,
            mut cache_layers,
            batch_dim,
            sampling_start,
            sampling_length,
            rope_max_sequence_length,
            rope_dim,
            #[cfg(feature = "tracing")]
            trace,
        } = args;
        #[cfg(feature = "tracing")]
        let mut trace = trace;

        let mut shortcut =
            encoder.allocate_scratch(main.as_buffer_range().1.len()).map_err(DecoderError::BackendError)?;

        for layer in self.layers.iter() {
            let rope_type = layer.rope_type();
            let rope_cosines = rope_type.and_then(|rope_type| shared_buffers.rope_cosines(rope_type));
            let rope_sines = rope_type.and_then(|rope_type| shared_buffers.rope_sines(rope_type));
            let attention_sinks = shared_buffers.attention_sinks(layer.layer_index);
            #[cfg(feature = "tracing")]
            let layer_trace = trace.as_deref_mut().map(|trace| &mut trace.layer_results[layer.layer_index]);

            let cache_layer = cache_layers.as_deref_mut().map(|cache_layers| &mut cache_layers.data[layer.layer_index]);
            main = layer
                .encode(
                    LayerArguments {
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
                        cache_layer,
                        #[cfg(feature = "tracing")]
                        trace: layer_trace,
                    },
                    parameters,
                    main,
                    &mut shortcut,
                    encoder,
                )
                .map_err(DecoderError::BackendError)?;
        }

        Ok((main, shortcut))
    }

    pub fn encode_prefill(
        &self,
        args: DecoderArguments<B>,
        parameters: &EncodingParameters,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, DecoderError<B>> {
        let main = self.embed.encode_lookup(args.token_ids, args.batch_dim, args.activation_data_type, encoder)?;
        let (main, _) = self.run_layers(args, main, parameters, encoder)?;
        Ok(main)
    }

    pub fn encode_decode(
        &self,
        args: DecoderArguments<B>,
        input: DecoderDecodeInput<B>,
        logits: Allocation<B>,
        hidden_capture: Option<&mut Allocation<B>>,
        parameters: &EncodingParameters,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, DecoderError<B>> {
        let sampling_start = args.sampling_start;
        let sampling_length = args.sampling_length;
        let batch_dim = args.batch_dim;
        #[cfg(feature = "tracing")]
        let mut trace = args.trace;
        let main = match input {
            DecoderDecodeInput::TokenIds => {
                self.embed.encode_lookup(args.token_ids, args.batch_dim, args.activation_data_type, encoder)?
            },
            DecoderDecodeInput::Embeddings(main) => main,
        };
        let (main, mut shortcut) = self.run_layers(
            DecoderArguments {
                activation_data_type: args.activation_data_type,
                token_ids: args.token_ids,
                token_positions: args.token_positions,
                token_parents: args.token_parents,
                token_subtrie_ranges: args.token_subtrie_ranges,
                shared_buffers: args.shared_buffers,
                cache_layers: args.cache_layers,
                batch_dim: args.batch_dim,
                sampling_start,
                sampling_length,
                rope_max_sequence_length: args.rope_max_sequence_length,
                rope_dim: args.rope_dim,
                #[cfg(feature = "tracing")]
                trace: trace.as_deref_mut(),
            },
            main,
            parameters,
            encoder,
        )?;

        let output_norm = self
            .norm
            .encode(&main, sampling_start, sampling_length, Some(&mut shortcut), encoder)
            .map_err(DecoderError::BackendError)?;
        if let Some(hidden_capture) = hidden_capture {
            let row_size_bytes = shortcut.as_buffer_range().1.len() / batch_dim;
            let input_offset = (batch_dim - 1) * row_size_bytes;
            encoder.encode_copy(&shortcut, input_offset..input_offset + row_size_bytes, hidden_capture, ..);
        }
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace.as_deref_mut() {
            encoder.encode_copy(&output_norm, .., &mut trace.output_norm, ..);
        }

        let mut logits = logits;
        self.embed.encode_readout(sampling_length, 0, &output_norm, &mut logits, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace.as_deref_mut() {
            encoder.encode_copy(&logits, .., &mut trace.logits, ..);
        }
        Ok(logits)
    }
}
