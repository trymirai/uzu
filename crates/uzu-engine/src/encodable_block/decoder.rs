use std::{range::Range, sync::Arc};

use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{
        EncodableBlock,
        batch_topology::BatchTopology,
        embedding::{
            EmbeddingError, EmbeddingLookup, EmbeddingLookupInput, EmbeddingReadout, EmbeddingReadoutInput,
            EmbeddingResource,
        },
        linear::Gather,
        logit_transform::{LogitTransform, LogitTransformInput},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        per_layer_embedding::{PerLayerEmbedding, PerLayerEmbeddingError},
        transformer::{Transformer, TransformerNewError, TransformerState},
    },
    parameters::ParameterTree,
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
}

pub struct Decoder<B: Backend> {
    embedding_lookup: EmbeddingLookup<B>,
    embedding_readout: EmbeddingReadout<B>,
    logit_transform: Option<LogitTransform<B>>,
    embedding_norm: Option<Normalization<B>>,
    per_layer_embedding: Option<PerLayerEmbedding<B>>,
    transformer: Transformer<B>,
}

pub struct DecoderEncodeOutput<B: Backend> {
    pub logits: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
    pub final_hidden: Option<B::ScratchBuffer>,
}

impl<B: Backend> Decoder<B> {
    pub fn embedding_lookup(&self) -> &EmbeddingLookup<B> {
        &self.embedding_lookup
    }

    pub fn embedding_readout(&self) -> &EmbeddingReadout<B> {
        &self.embedding_readout
    }

    pub fn new(
        context: &B::Context,
        config: &DecoderConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DecoderError<B>> {
        let embedding_tree = parameter_tree.subtree("embedding");
        let embedding = Arc::new(EmbeddingResource::load_input(
            &config.embedding_config,
            &embedding_tree,
            config.vocab_size,
            config.transformer_config.model_dim,
            data_type,
        )?);
        let (embedding_readout, readout_input_hadamard_factors) =
            EmbeddingReadout::new(context, &config.embedding_config, &embedding_tree, &embedding)?;
        let logit_transform =
            LogitTransform::new(context, &config.embedding_config, data_type).map_err(DecoderError::Backend)?;
        let embedding_lookup =
            EmbeddingLookup::new(context, embedding, config.embedding_config.input_scale().unwrap_or(1.0))
                .map_err(DecoderError::Backend)?;

        let embedding_norm = config
            .embedding_norm_config
            .as_ref()
            .map(|norm_config| {
                Normalization::new(
                    config.transformer_config.model_dim,
                    None,
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    data_type,
                    norm_config,
                    &parameter_tree.subtree("embedding_norm"),
                    context,
                )
            })
            .transpose()?;

        let per_layer_embedding = if let Some(ple_config) = &config.ple_model_config {
            assert_eq!(
                ple_config.num_layers,
                config.transformer_config.layer_configs.len() as u32,
                "per-layer embedding num_layers must match transformer layer count"
            );
            Some(PerLayerEmbedding::new(
                context,
                ple_config,
                config.transformer_config.model_dim,
                data_type,
                &parameter_tree.subtree("per_layer_embedding"),
            )?)
        } else {
            None
        };

        let transformer = Transformer::new(
            context,
            readout_input_hadamard_factors,
            data_type,
            &config.transformer_config,
            &parameter_tree.subtree("transformer"),
        )?;

        Ok(Self {
            embedding_lookup,
            embedding_readout,
            logit_transform,
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
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        command_buffer.push_debug_group("decoder");

        let embedded = self
            .embedding_lookup
            .encode(
                EmbeddingLookupInput {
                    token_ids,
                    batch_dim: batch_dim.size(),
                },
                command_buffer,
            )
            .map_err(DecoderError::Backend)?;
        let embedded = if let Some(embedding_norm) = &self.embedding_norm {
            embedding_norm
                .encode(&embedded, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(DecoderError::Backend)?
        } else {
            embedded
        };

        let per_layer_inputs = if let Some(per_layer_embedding) = &self.per_layer_embedding {
            Some(
                per_layer_embedding
                    .encode(token_ids, &embedded, batch_dim.size(), command_buffer)
                    .map_err(DecoderError::Backend)?,
            )
        } else {
            None
        };

        let transformer_output = self
            .transformer
            .encode(
                embedded,
                per_layer_inputs.as_ref(),
                batch_dim,
                output_range,
                hidden_feature_layer_indices,
                Some(state),
                command_buffer,
            )
            .map_err(DecoderError::Backend)?;

        let logits = if let Some(output_range) = output_range {
            let output = transformer_output.output.as_ref().expect("decoder output range requires transformer output");
            let output_rows = output_range.end - output_range.start;
            let mut logits = self
                .embedding_readout
                .encode(
                    EmbeddingReadoutInput {
                        hidden: output,
                        batch_dim: output_rows,
                        gather: None::<Gather<&B::ScratchBuffer>>,
                    },
                    command_buffer,
                )
                .map_err(DecoderError::Backend)?;
            if let Some(logit_transform) = &self.logit_transform {
                let Ok(()) = logit_transform.encode(
                    LogitTransformInput {
                        logits: &mut logits,
                        length: output_rows * self.embedding_readout.vocab_size(),
                    },
                    command_buffer,
                );
            }
            Some(logits)
        } else {
            None
        };
        let final_hidden = if hidden_feature_layer_indices.is_none() {
            None
        } else {
            transformer_output.output
        };

        command_buffer.pop_debug_group();

        Ok(DecoderEncodeOutput {
            logits,
            hidden_features: transformer_output.hidden_features,
            final_hidden,
        })
    }
}
