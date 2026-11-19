use std::range::Range;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        per_layer_embedding::{PerLayerEmbedding, PerLayerEmbeddingError},
        transformer::{Transformer, TransformerEncodeError, TransformerNewError, TransformerPass, TransformerState},
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum DecoderError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Per-layer embedding error: {0}")]
    PerLayerEmbedding(#[from] PerLayerEmbeddingError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
    #[error("Transformer encoding error: {0}")]
    TransformerEncode(#[from] TransformerEncodeError<B>),
}

pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    embedding_norm: Option<Normalization<B>>,
    per_layer_embedding: Option<PerLayerEmbedding<B>>,
    transformer: Transformer<B>,
}

pub struct DecoderEncodeOutput<B: Backend> {
    pub logits: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
    pub final_hidden: Option<B::ScratchBuffer>,
}

/// See [`Decoder::begin`].
pub struct DecoderPass<B: Backend> {
    transformer: TransformerPass<B>,
    per_layer_inputs: Option<B::ScratchBuffer>,
}

impl<B: Backend> Decoder<B> {
    pub(crate) fn embedding(&self) -> &Embedding<B> {
        &self.embedding
    }

    pub async fn new(
        data_type: DataType,
        config: &DecoderConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, DecoderError<B>> {
        let (embedding, readout_input_hadamard_factors) = Embedding::new(
            config.vocab_size,
            config.transformer_config.model_dim,
            data_type,
            &config.embedding_config,
            &parameter_tree.subtree("embedding"),
            context,
        )
        .await?;

        let embedding_norm = if let Some(norm_config) = &config.embedding_norm_config {
            Some(
                Normalization::new(
                    config.transformer_config.model_dim,
                    data_type,
                    data_type,
                    None,
                    ShortcutMode::None,
                    norm_config,
                    &parameter_tree.subtree("embedding_norm"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        let per_layer_embedding = if let Some(ple_config) = &config.ple_model_config {
            assert_eq!(
                ple_config.num_layers,
                config.transformer_config.layer_configs.len() as u32,
                "per-layer embedding num_layers must match transformer layer count"
            );
            Some(
                PerLayerEmbedding::new(
                    config.transformer_config.model_dim,
                    data_type,
                    ple_config,
                    &parameter_tree.subtree("per_layer_embedding"),
                    context,
                )
                .await?,
            )
        } else {
            None
        };

        let transformer = Transformer::new(
            data_type,
            data_type,
            readout_input_hadamard_factors,
            &config.transformer_config,
            &parameter_tree.subtree("transformer"),
            context,
        )
        .await?;

        Ok(Self {
            embedding,
            embedding_norm,
            per_layer_embedding,
            transformer,
        })
    }

    pub fn speculation_supported(&self) -> bool {
        self.transformer.speculation_supported()
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.transformer.max_context_length()
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.transformer.prefill_cache_skips_trailing_layers()
    }

    pub fn transformer(&self) -> &Transformer<B> {
        &self.transformer
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        self.transformer.create_empty_state(max_context_length, context)
    }

    pub fn encode(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology<B>,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("decoder");

        let pass = self.begin(token_ids, batch_dim, output_range, hidden_feature_layer_indices, command_buffer)?;
        self.finish(pass, batch_dim, state, command_buffer)
    }

    pub fn begin(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology<B>,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderPass<B>, DecoderError<B>> {
        let embedded = self.embedding.encode_lookup(token_ids, batch_dim.size(), command_buffer)?;
        let embedded = if let Some(embedding_norm) = &self.embedding_norm {
            embedding_norm
                .encode(&embedded, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(DecoderError::Backend)?
        } else {
            embedded
        };

        let per_layer_inputs = if let Some(per_layer_embedding) = &self.per_layer_embedding {
            Some(per_layer_embedding.encode(token_ids, &embedded, batch_dim.size(), command_buffer)?)
        } else {
            None
        };

        let transformer =
            self.transformer.begin(embedded, batch_dim, output_range, hidden_feature_layer_indices, command_buffer)?;

        Ok(DecoderPass {
            transformer,
            per_layer_inputs,
        })
    }

    pub fn encode_layers(
        &self,
        pass: DecoderPass<B>,
        until: usize,
        batch_dim: &BatchTopology<B>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderPass<B>, DecoderError<B>> {
        let DecoderPass {
            transformer,
            per_layer_inputs,
        } = pass;
        let transformer = self.transformer.encode_layers(
            transformer,
            until,
            per_layer_inputs.as_ref(),
            batch_dim,
            Some(state),
            command_buffer,
        )?;

        Ok(DecoderPass {
            transformer,
            per_layer_inputs,
        })
    }

    pub fn finish(
        &self,
        pass: DecoderPass<B>,
        batch_dim: &BatchTopology<B>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        let DecoderPass {
            transformer,
            per_layer_inputs,
        } = pass;
        let output_range = transformer.output_range();
        let transformer_output =
            self.transformer.finish(transformer, per_layer_inputs.as_ref(), batch_dim, Some(state), command_buffer)?;

        let logits = if let Some(output_range) = output_range {
            let output = transformer_output.output.as_ref().expect("decoder output range requires transformer output");
            Some(self.embedding.encode_readout(
                output,
                output_range.end - output_range.start,
                self.embedding.vocab_size(),
                None::<&B::ScratchBuffer>,
                true,
                command_buffer,
            )?)
        } else {
            None
        };

        Ok(DecoderEncodeOutput {
            logits,
            final_hidden: transformer_output.hidden_features.as_ref().and(transformer_output.output),
            hidden_features: transformer_output.hidden_features,
        })
    }

    pub fn has_ngram_embeddings(&self) -> bool {
        self.transformer.has_ngram_embeddings()
    }

    pub fn first_ngram_layer_index(&self) -> Option<usize> {
        self.transformer.first_ngram_layer_index()
    }

    pub fn prepare_ngram_rows(
        &self,
        state: &mut TransformerState<B>,
        token_ids: &[u32],
        parents: Option<&[i32]>,
    ) -> std::io::Result<()> {
        self.transformer.prepare_ngram_rows(state, token_ids, parents)
    }
}
