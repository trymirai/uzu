//! Decoder executables - combines embedding, layers, normalization, and readout.

use std::rc::Rc;

use thiserror::Error;

#[cfg(feature = "tracing")]
use crate::forward_pass::traces::ActivationTrace;
use crate::{
    backends::common::{Allocation, AsBufferRangeRef, Backend, Encoder},
    config::{decoder::DecoderConfig, rope::AnyRoPEConfig},
    data_type::DataType,
    encodable_block::{
        Embedding, LayerArguments, LayerExecutables, LayerRopeKind, PerLayerEmbedding, PostLayerScalar,
        PrecalculatedRope, QkUnpack, RMSNorm, RMSNormError, Rope, embedding::EmbeddingError,
        layer::LayerExecutablesError,
    },
    forward_pass::{cache_layers::CacheLayers, model_shape::ModelShape, rope::precalculate_rope},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum DecoderError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Layer error: {0}")]
    LayerError(#[from] LayerExecutablesError<B>),
    #[error("RMSNorm error: {0}")]
    RMSNormError(#[from] RMSNormError<B>),
    #[error("Invalid decode input: {0}")]
    InvalidDecodeInput(&'static str),
}

/// Full decoder executable with all layers and components.
pub struct Decoder<B: Backend> {
    pub embed: Embedding<B>,
    pub rope_configs: Box<[AnyRoPEConfig]>,
    pub layers: Box<[LayerExecutables<B>]>,
    pub norm: RMSNorm<B>,
    pub per_layer_embedding: Option<PerLayerEmbedding<B>>,
}

pub struct DecoderArguments<'a, B: Backend> {
    pub token_positions: &'a [usize],
    pub token_parents: &'a Allocation<B>,
    pub token_subtrie_ranges: Option<&'a Allocation<B>>,
    pub cache_layers: Option<&'a mut CacheLayers<B>>,
    pub batch_dim: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    #[cfg(feature = "tracing")]
    pub trace: Option<&'a mut ActivationTrace<B>>,
}

pub enum DecoderDecodeInput<'a, B: Backend> {
    TokenIds(&'a Allocation<B>),
    #[cfg(metal_backend)]
    Embeddings(Allocation<B>),
}

impl<B: Backend> Decoder<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B>,
        model_shape: &ModelShape,
    ) -> Result<Self, DecoderError<B>> {
        let embedding_weight_loader = root_weight_loader.subtree("embedding")?;

        let (embed, readout_input_hadamard_factors) = Embedding::new(
            context,
            decoder_config.vocab_size as u32,
            decoder_config.transformer_config.model_dim as u32,
            &decoder_config.embedding_config,
            &embedding_weight_loader,
            model_shape,
        )?;

        let (rope_configs, layers, norm) = Self::build_transformer_layers_and_norm(
            context,
            decoder_config,
            root_weight_loader,
            "transformer",
            readout_input_hadamard_factors,
            model_shape,
        )?;

        let per_layer_embedding =
            Self::build_per_layer_embedding(context, decoder_config, root_weight_loader, model_shape.data_type);

        Ok(Self {
            embed,
            rope_configs,
            layers,
            norm,
            per_layer_embedding,
        })
    }

    /// Used by TTS models whose token lookup weights live outside the
    /// standard decoder subtree. The TTS runtime owns its separate readout.
    #[cfg(metal_backend)]
    pub fn new_with_embedding_subtree(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        model_shape: &ModelShape,
    ) -> Result<Self, DecoderError<B>> {
        let embedding_weight_loader = root_weight_loader.subtree(embedding_subtree)?;

        let (embed, readout_input_hadamard_factors) = Embedding::new(
            context,
            decoder_config.vocab_size as u32,
            decoder_config.transformer_config.model_dim as u32,
            &decoder_config.embedding_config,
            &embedding_weight_loader,
            model_shape,
        )?;

        let (rope_configs, layers, norm) = Self::build_transformer_layers_and_norm(
            context,
            decoder_config,
            root_weight_loader,
            transformer_subtree,
            readout_input_hadamard_factors,
            model_shape,
        )?;

        let per_layer_embedding =
            Self::build_per_layer_embedding(context, decoder_config, root_weight_loader, model_shape.data_type);

        Ok(Self {
            embed,
            rope_configs,
            layers,
            norm,
            per_layer_embedding,
        })
    }

    pub fn build_transformer_layers_and_norm(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B>,
        transformer_subtree: &str,
        output_norm_hadamard_factors: Option<Allocation<B>>,
        model_shape: &ModelShape,
    ) -> Result<(Box<[AnyRoPEConfig]>, Box<[LayerExecutables<B>]>, RMSNorm<B>), DecoderError<B>> {
        let decoder_weight_loader = root_weight_loader.subtree(transformer_subtree)?;

        let tf = &decoder_config.transformer_config;
        let mut rope_configs = Vec::<(AnyRoPEConfig, usize)>::new();
        let layer_rope_kinds: Box<[LayerRopeKind]> = tf
            .layer_configs
            .iter()
            .map(|layer_config| {
                if layer_config.mixer_config.as_attention().is_none() {
                    return LayerRopeKind::NoKernel;
                }
                let Some(rope_config) = &layer_config.rope_config else {
                    return LayerRopeKind::NoKernel;
                };
                let head_dim = *rope_config.head_dim();
                let index = rope_configs
                    .iter()
                    .position(|(existing_config, existing_head_dim)| {
                        existing_config == rope_config && *existing_head_dim == head_dim
                    })
                    .unwrap_or_else(|| {
                        rope_configs.push((rope_config.clone(), head_dim));
                        rope_configs.len() - 1
                    });
                LayerRopeKind::Indexed(index)
            })
            .collect();

        let rope = Rc::new(Rope::<B>::new(context, model_shape, false).map_err(DecoderError::BackendError)?);
        // Built only when some layer projects queries only (reuses an earlier layer's KV cache).
        let rope_query_only = if tf.layer_configs.iter().any(|l| l.kv_source_layer_index.is_some()) {
            Some(Rc::new(Rope::<B>::new(context, model_shape, true).map_err(DecoderError::BackendError)?))
        } else {
            None
        };
        let qk_unpack =
            Rc::new(QkUnpack::<B>::new(context, model_shape.data_type).map_err(DecoderError::BackendError)?);

        let layers = tf
            .layer_configs
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| {
                let layer_loader = decoder_weight_loader.subtree(&format!("layers.{}", layer_index))?;
                let layer_rope = if layer_config.kv_source_layer_index.is_some() {
                    rope_query_only.as_ref().expect("query-only rope is built whenever a layer shares KV")
                } else {
                    &rope
                };

                LayerExecutables::new(
                    context,
                    tf,
                    layer_config,
                    layer_index,
                    layer_rope_kinds[layer_index],
                    &layer_loader,
                    layer_rope,
                    &qk_unpack,
                    model_shape.data_type,
                )
                .map_err(DecoderError::LayerError)
            })
            .collect::<Result<Vec<_>, DecoderError<B>>>()?;

        let norm_block = RMSNorm::new(
            context,
            model_shape.data_type,
            model_shape.model_dim(),
            tf.output_norm_config.clone(),
            &decoder_weight_loader.subtree("output_norm")?,
            output_norm_hadamard_factors,
            true,
            true,
            PostLayerScalar::None,
        )?;

        Ok((rope_configs.into_iter().map(|(config, _)| config).collect(), layers.into_boxed_slice(), norm_block))
    }

    fn build_per_layer_embedding(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B>,
        data_type: DataType,
    ) -> Option<PerLayerEmbedding<B>> {
        decoder_config.ple_model_config.as_ref().map(|ple_config| {
            let layer_count = decoder_config.transformer_config.layer_configs.len();
            assert_eq!(
                ple_config.num_layers, layer_count,
                "per-layer embedding num_layers must match transformer layer count"
            );
            let ple_weight_loader =
                root_weight_loader.subtree("per_layer_embedding").expect("Failed to get per_layer_embedding subtree");
            PerLayerEmbedding::new(
                context,
                ple_config,
                decoder_config.transformer_config.model_dim,
                data_type,
                &ple_weight_loader,
            )
            .expect("Failed to create per-layer embedding")
        })
    }

    fn encode_per_layer_inputs(
        &self,
        token_ids: &Allocation<B>,
        inner_features: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Option<Allocation<B>>, DecoderError<B>> {
        if let Some(ple) = &self.per_layer_embedding {
            let output =
                ple.encode(token_ids, inner_features, batch_dim, encoder).map_err(DecoderError::BackendError)?;
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    fn run_layers(
        &self,
        args: DecoderArguments<B>,
        mut main: Allocation<B>,
        per_layer_inputs: Option<&Allocation<B>>,
        layer_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), DecoderError<B>> {
        let DecoderArguments {
            token_positions,
            token_parents,
            token_subtrie_ranges,
            mut cache_layers,
            batch_dim,
            sampling_start,
            sampling_length,
            #[cfg(feature = "tracing")]
            trace,
        } = args;
        #[cfg(feature = "tracing")]
        let mut trace = trace;

        let mut shortcut =
            encoder.allocate_scratch(main.as_buffer_range_ref().range().len()).map_err(DecoderError::BackendError)?;
        let ropes = self
            .rope_configs
            .iter()
            .map(|rope_config| {
                let (sines, cosines) = precalculate_rope(rope_config, token_positions);
                let mut sines_allocation = encoder.allocate_constant(sines.len() * std::mem::size_of::<f32>())?;
                sines_allocation.copyin(sines.as_ref());
                let mut cosines_allocation = encoder.allocate_constant(cosines.len() * std::mem::size_of::<f32>())?;
                cosines_allocation.copyin(cosines.as_ref());
                Ok(PrecalculatedRope {
                    cosines: cosines_allocation,
                    sines: sines_allocation,
                    dim: *rope_config.head_dim(),
                })
            })
            .collect::<Result<Box<[_]>, B::Error>>()
            .map_err(DecoderError::BackendError)?;

        for layer in self.layers.iter().take(layer_count) {
            #[cfg(feature = "tracing")]
            let layer_trace = trace.as_deref_mut().map(|trace| &mut trace.layer_results[layer.layer_index]);

            let cache_access =
                cache_layers.as_deref_mut().map(|cache_layers| cache_layers.cache_for_layer(layer.layer_index));
            main = layer
                .encode(
                    LayerArguments {
                        batch_dim,
                        token_parents,
                        token_subtrie_ranges,
                        rope: layer.rope_kind.get(&ropes),
                        per_layer_inputs,
                        sampling_start,
                        sampling_length,
                        cache_access,
                        #[cfg(feature = "tracing")]
                        trace: layer_trace,
                    },
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
        token_ids: &Allocation<B>,
        layer_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, DecoderError<B>> {
        let main = self.embed.encode_lookup(token_ids, args.batch_dim, encoder)?;
        let per_layer_inputs = self.encode_per_layer_inputs(token_ids, &main, args.batch_dim, encoder)?;
        let (main, _) = self.run_layers(args, main, per_layer_inputs.as_ref(), layer_count, encoder)?;
        Ok(main)
    }

    pub fn encode_decode<'input>(
        &self,
        args: DecoderArguments<B>,
        input: DecoderDecodeInput<'input, B>,
        hidden_capture: Option<&mut Allocation<B>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, DecoderError<B>> {
        let sampling_start = args.sampling_start;
        let sampling_length = args.sampling_length;
        let batch_dim = args.batch_dim;
        #[cfg(feature = "tracing")]
        let mut trace = args.trace;
        let (main, token_ids) = match input {
            DecoderDecodeInput::TokenIds(token_ids) => {
                let main = self.embed.encode_lookup(token_ids, args.batch_dim, encoder)?;
                (main, Some(token_ids))
            },
            #[cfg(metal_backend)]
            DecoderDecodeInput::Embeddings(main) => (main, None),
        };
        let per_layer_inputs = match token_ids {
            Some(token_ids) => self.encode_per_layer_inputs(token_ids, &main, batch_dim, encoder)?,
            None if self.per_layer_embedding.is_some() => {
                return Err(DecoderError::InvalidDecodeInput("per-layer embedding requires token ids"));
            },
            None => None,
        };
        let (main, mut shortcut) = self.run_layers(
            DecoderArguments {
                token_positions: args.token_positions,
                token_parents: args.token_parents,
                token_subtrie_ranges: args.token_subtrie_ranges,
                cache_layers: args.cache_layers,
                batch_dim: args.batch_dim,
                sampling_start,
                sampling_length,
                #[cfg(feature = "tracing")]
                trace: trace.as_deref_mut(),
            },
            main,
            per_layer_inputs.as_ref(),
            self.layers.len(),
            encoder,
        )?;

        let output_norm = self
            .norm
            .encode(&main, sampling_start, sampling_length, Some(&mut shortcut), encoder)
            .map_err(DecoderError::BackendError)?;
        if let Some(hidden_capture) = hidden_capture {
            let row_size_bytes = shortcut.as_buffer_range_ref().range().len() / batch_dim;
            let input_offset = (batch_dim - 1) * row_size_bytes;
            encoder.encode_copy(&shortcut, input_offset..input_offset + row_size_bytes, hidden_capture, ..);
        }
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace.as_deref_mut() {
            encoder.encode_copy(&output_norm, .., trace.output_norm.allocation_mut(), ..);
        }

        let logits = self.embed.encode_readout(sampling_length, &output_norm, encoder)?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace {
            encoder.encode_copy(&logits, .., trace.logits.allocation_mut(), ..);
        }
        Ok(logits)
    }
}
