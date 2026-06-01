use std::ops::Range;

use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{
        embedding::{Embedding, EmbeddingError},
        mixer::MixerTokenTopology,
        per_layer_embedding::{PerLayerEmbedding, PerLayerEmbeddingError},
        transformer::{Transformer, TransformerNewError, TransformerState},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum DecoderError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
    #[error("Per-layer embedding error: {0}")]
    PerLayerEmbedding(#[from] PerLayerEmbeddingError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
}

pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    per_layer_embedding: Option<PerLayerEmbedding<B>>,
    transformer: Transformer<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn new(
        context: &B::Context,
        config: &DecoderConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DecoderError<B>> {
        let (embedding, readout_input_hadamard_factors) = Embedding::new(
            context,
            config.vocab_size as u32,
            config.transformer_config.model_dim as u32,
            &config.embedding_config,
            &parameter_tree.subtree("embedding")?,
            data_type,
        )?;

        let per_layer_embedding = if let Some(ple_config) = &config.ple_model_config {
            assert_eq!(
                ple_config.num_layers,
                config.transformer_config.layer_configs.len(),
                "per-layer embedding num_layers must match transformer layer count"
            );
            Some(PerLayerEmbedding::new(
                context,
                ple_config,
                config.transformer_config.model_dim,
                data_type,
                &parameter_tree.subtree("per_layer_embedding")?,
            )?)
        } else {
            None
        };

        let transformer = Transformer::new(
            context,
            readout_input_hadamard_factors,
            data_type,
            &config.transformer_config,
            &parameter_tree.subtree("transformer")?,
        )?;

        Ok(Self {
            embedding,
            per_layer_embedding,
            transformer,
        })
    }

    pub fn trie_supported(&self) -> bool {
        self.transformer.trie_supported()
    }

    pub fn max_context_length(&self) -> Option<usize> {
        self.transformer.max_context_length()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        self.transformer.create_empty_state(max_context_length, context)
    }

    pub fn encode(
        &self,
        token_ids: &Allocation<B>,
        batch_dim: usize,
        output_range: Option<Range<usize>>,
        token_topology: &MixerTokenTopology<B>,
        state: &mut TransformerState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Option<Allocation<B>>, DecoderError<B>> {
        let embedded = self.embedding.encode_lookup(token_ids, batch_dim, encoder)?;

        let per_layer_inputs = if let Some(per_layer_embedding) = &self.per_layer_embedding {
            Some(per_layer_embedding.encode(token_ids, &embedded, batch_dim, encoder).map_err(DecoderError::Backend)?)
        } else {
            None
        };

        let hidden = self
            .transformer
            .encode(
                embedded,
                per_layer_inputs.as_ref(),
                batch_dim,
                output_range.clone(),
                token_topology,
                Some(state),
                encoder,
            )
            .map_err(DecoderError::Backend)?;

        let Some(output_range) = output_range else {
            return Ok(None);
        };

        let logits = self.embedding.encode_readout(output_range.len(), &hidden.unwrap(), encoder)?;

        Ok(Some(logits))
    }
}
