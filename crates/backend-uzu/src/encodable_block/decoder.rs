//! Decoder executables - combines embedding, layers, normalization, and readout.

use std::rc::Rc;

use thiserror::Error;

#[cfg(feature = "tracing")]
use crate::forward_pass::traces::ActivationTrace;
use crate::{
    backends::common::{Allocation, AsBufferRangeRef, Backend, Encoder},
    config::decoder::DecoderConfig,
    encodable_block::{
        Embedding, LayerArguments, LayerExecutables, PostLayerScalar, QkUnpack, RMSNorm, RMSNormError, Rope,
        embedding::EmbeddingError, layer::LayerExecutablesError,
    },
    forward_pass::{cache_layers::CacheLayers, model_shape::ModelShape, state::SharedBuffers},
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
}

/// Full decoder executable with all layers and components.
pub struct Decoder<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Box<[LayerExecutables<B>]>,
    pub norm: RMSNorm<B>,
}

pub struct DecoderArguments<'a, B: Backend> {
    pub token_positions: &'a Allocation<B>,
    pub token_parents: &'a Allocation<B>,
    pub token_subtrie_ranges: Option<&'a Allocation<B>>,
    pub shared_buffers: &'a SharedBuffers<B>,
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

        let (layers, norm) = Self::build_transformer_layers_and_norm(
            context,
            decoder_config,
            root_weight_loader,
            "transformer",
            readout_input_hadamard_factors,
            model_shape,
        )?;

        Ok(Self {
            embed,
            layers,
            norm,
        })
    }

    /// Used by models whose token lookup weights and logits readout weights
    #[cfg(metal_backend)]
    pub fn new_with_embedding_and_readout_subtrees(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
        model_shape: &ModelShape,
    ) -> Result<Self, DecoderError<B>> {
        let embedding_weight_loader = root_weight_loader.subtree(embedding_subtree)?;
        let readout_weight_loader = root_weight_loader.subtree(readout_subtree)?;

        let (embed, readout_input_hadamard_factors) = Embedding::new_with_lookup_and_readout_trees(
            context,
            decoder_config.vocab_size as u32,
            decoder_config.transformer_config.model_dim as u32,
            &decoder_config.embedding_config,
            &embedding_weight_loader,
            &readout_weight_loader,
            model_shape,
        )?;

        let (layers, norm) = Self::build_transformer_layers_and_norm(
            context,
            decoder_config,
            root_weight_loader,
            transformer_subtree,
            readout_input_hadamard_factors,
            model_shape,
        )?;

        Ok(Self {
            embed,
            layers,
            norm,
        })
    }

    pub(crate) fn build_transformer_layers_and_norm(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        root_weight_loader: &ParameterTree<B>,
        transformer_subtree: &str,
        output_norm_hadamard_factors: Option<Allocation<B>>,
        model_shape: &ModelShape,
    ) -> Result<(Box<[LayerExecutables<B>]>, RMSNorm<B>), DecoderError<B>> {
        let decoder_weight_loader = root_weight_loader.subtree(transformer_subtree)?;

        let tf = &decoder_config.transformer_config;
        let rope = Rc::new(Rope::<B>::new(context, model_shape).map_err(DecoderError::BackendError)?);
        let qk_unpack =
            Rc::new(QkUnpack::<B>::new(context, model_shape.data_type).map_err(DecoderError::BackendError)?);

        let layers = tf
            .layer_configs
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| {
                let layer_loader = decoder_weight_loader.subtree(&format!("layers.{}", layer_index))?;

                LayerExecutables::new(
                    context,
                    tf,
                    layer_config,
                    layer_index,
                    &layer_loader,
                    &rope,
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

        Ok((layers.into_boxed_slice(), norm_block))
    }

    fn run_layers(
        &self,
        args: DecoderArguments<B>,
        mut main: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), DecoderError<B>> {
        let DecoderArguments {
            token_positions,
            token_parents,
            token_subtrie_ranges,
            shared_buffers,
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

        for layer in self.layers.iter() {
            let rope_buffers = shared_buffers.rope_buffers_for_layer(layer.layer_index);
            #[cfg(feature = "tracing")]
            let layer_trace = trace.as_deref_mut().map(|trace| &mut trace.layer_results[layer.layer_index]);

            let cache_access =
                cache_layers.as_deref_mut().map(|cache_layers| cache_layers.cache_for_layer(layer.layer_index));
            main = layer
                .encode(
                    LayerArguments {
                        batch_dim,
                        token_positions,
                        token_parents,
                        token_subtrie_ranges,
                        rope_buffers,
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
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, DecoderError<B>> {
        let main = self.embed.encode_lookup(token_ids, args.batch_dim, encoder)?;
        let (main, _) = self.run_layers(args, main, encoder)?;
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
        let main = match input {
            DecoderDecodeInput::TokenIds(token_ids) => self.embed.encode_lookup(token_ids, args.batch_dim, encoder)?,
            #[cfg(metal_backend)]
            DecoderDecodeInput::Embeddings(main) => main,
        };
        let (main, mut shortcut) = self.run_layers(
            DecoderArguments {
                token_positions: args.token_positions,
                token_parents: args.token_parents,
                token_subtrie_ranges: args.token_subtrie_ranges,
                shared_buffers: args.shared_buffers,
                cache_layers: args.cache_layers,
                batch_dim: args.batch_dim,
                sampling_start,
                sampling_length,
                #[cfg(feature = "tracing")]
                trace: trace.as_deref_mut(),
            },
            main,
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
        if let Some(trace) = trace.as_deref_mut() {
            encoder.encode_copy(&logits, .., trace.logits.allocation_mut(), ..);
        }
        Ok(logits)
    }
}
