use std::ops::Range;

use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        per_layer_embedding::{PerLayerEmbedding, PerLayerEmbeddingError},
        transformer::{Transformer, TransformerNewError, TransformerState},
    },
    parameters::ParameterTree,
    trace::{Array, DecoderTap, DecoderTapRequest},
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
    embedding: Embedding<B>,
    embedding_norm: Option<Normalization<B>>,
    per_layer_embedding: Option<PerLayerEmbedding<B>>,
    transformer: Transformer<B>,
}

pub struct DecoderEncodeOutput<B: Backend> {
    pub logits: Option<Allocation<B>>,
    pub tap: DecoderTap<B>,
    #[allow(dead_code)]
    pub hidden_features: Option<Box<[Allocation<B>]>>,
    #[allow(dead_code)]
    pub final_hidden: Option<Allocation<B>>,
}

impl<B: Backend> Decoder<B> {
    pub(crate) fn embedding(&self) -> &Embedding<B> {
        &self.embedding
    }

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
            &parameter_tree.subtree("embedding"),
            data_type,
        )?;

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
                config.transformer_config.layer_configs.len(),
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
            embedding,
            embedding_norm,
            per_layer_embedding,
            transformer,
        })
    }

    pub fn speculation_supported(&self) -> bool {
        self.transformer.speculation_supported()
    }

    pub fn max_context_length(&self) -> Option<usize> {
        self.transformer.max_context_length()
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.transformer.prefill_cache_skips_trailing_layers()
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
        batch_dim: &BatchTopology,
        output_range: Option<Range<usize>>,
        hidden_feature_layer_indices: Option<&[usize]>,
        state: &mut TransformerState<B>,
        tap_request: Option<&DecoderTapRequest>,
        encoder: &mut Encoder<B>,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        encoder.push_debug_group("decoder");

        let request = tap_request.unwrap_or(&DecoderTapRequest::NONE);
        let mut tap = DecoderTap::default();

        let embedded = self.embedding.encode_lookup(token_ids, batch_dim.size(), encoder)?;
        let embedded = if let Some(embedding_norm) = &self.embedding_norm {
            embedding_norm.encode(&embedded, 0, batch_dim.size(), None, encoder).map_err(DecoderError::Backend)?
        } else {
            embedded
        };
        if request.embedded {
            let shape = [1, batch_dim.size(), self.embedding.model_dim()];
            tap.embedded = Some(
                Array::capture(&embedded, &shape, self.embedding.data_type(), encoder)
                    .map_err(DecoderError::Backend)?,
            );
        }

        let per_layer_inputs = if let Some(per_layer_embedding) = &self.per_layer_embedding {
            Some(
                per_layer_embedding
                    .encode(token_ids, &embedded, batch_dim.size(), encoder)
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
                output_range.clone(),
                hidden_feature_layer_indices,
                Some(state),
                request.transformer.as_ref(),
                encoder,
            )
            .map_err(DecoderError::Backend)?;
        tap.transformer = request.transformer.is_some().then_some(transformer_output.tap);

        let logits = if let Some(output_range) = output_range {
            let output = transformer_output.output.as_ref().expect("decoder output range requires transformer output");
            let logits =
                self.embedding.encode_readout(output_range.len(), output, self.embedding.data_type(), encoder)?;
            if request.logits {
                let shape = [1, output_range.len(), self.embedding.vocab_size()];
                tap.logits = Some(
                    Array::capture(&logits, &shape, self.embedding.data_type(), encoder)
                        .map_err(DecoderError::Backend)?,
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

        encoder.pop_debug_group();

        Ok(DecoderEncodeOutput {
            logits,
            tap,
            hidden_features: transformer_output.hidden_features,
            final_hidden,
        })
    }
}
