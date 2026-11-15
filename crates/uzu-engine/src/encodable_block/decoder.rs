use std::range::Range;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        per_layer_embedding::{PerLayerEmbedding, PerLayerEmbeddingError},
        transformer::{Transformer, TransformerEncodeError, TransformerNewError, TransformerState},
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

impl<B: Backend> Decoder<B> {
    pub fn embedding(&self) -> &Embedding<B> {
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

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        self.transformer.create_empty_state(max_context_length, context)
    }

    pub fn model_dim(&self) -> u32 {
        self.transformer.model_dim()
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
        self.encode_multimodal(
            token_ids,
            None::<&B::ConstantBuffer>,
            None,
            None,
            0,
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            state,
            command_buffer,
        )
    }

    pub fn encode_multimodal(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        visual_features: Option<impl BufferRef<Backend = B>>,
        visual_slices: Option<&[(usize, usize)]>,
        mrope_positions: Option<&[[u32; 3]]>,
        mrope_delta: i64,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        self.encode_multimodal_with_logits(
            token_ids,
            visual_features,
            visual_slices,
            mrope_positions,
            mrope_delta,
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            true,
            state,
            command_buffer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_multimodal_with_logits(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        visual_features: Option<impl BufferRef<Backend = B>>,
        visual_slices: Option<&[(usize, usize)]>,
        mrope_positions: Option<&[[u32; 3]]>,
        mrope_delta: i64,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        compute_logits: bool,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("decoder");

        let mut embedded = self.embedding.encode_lookup(token_ids, batch_dim.size(), command_buffer)?;

        if let (Some(features), Some(slices)) = (visual_features, visual_slices) {
            let model_dim = self.transformer.model_dim() as usize;
            let bytes_per_token = model_dim * 2; // BF16 is 2 bytes
            let mut src_byte_offset = 0;
            for &(start_idx, count) in slices {
                let byte_len = count * bytes_per_token;
                let dst_byte_start = start_idx * bytes_per_token;
                let src_sub = features.subrange(src_byte_offset..src_byte_offset + byte_len);
                let dst_sub = embedded.subrange_mut(dst_byte_start..dst_byte_start + byte_len);
                command_buffer.encode_copy(src_sub, dst_sub);
                src_byte_offset += byte_len;
            }
        }

        self.encode_from_embedded(
            embedded,
            Some(token_ids),
            mrope_positions,
            mrope_delta,
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            compute_logits,
            state,
            command_buffer,
        )
    }

    /// Runs the decoder starting from precomputed token/multimodal embeddings (`[batch, model_dim]`).
    pub fn encode_precomputed_embeddings_with_logits(
        &self,
        embedded: B::ScratchBuffer,
        mrope_positions: Option<&[[u32; 3]]>,
        mrope_delta: i64,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        compute_logits: bool,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("decoder");
        self.encode_from_embedded(
            embedded,
            None::<&B::ConstantBuffer>,
            mrope_positions,
            mrope_delta,
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            compute_logits,
            state,
            command_buffer,
        )
    }

    fn encode_from_embedded(
        &self,
        embedded: B::ScratchBuffer,
        token_ids: Option<impl BufferRef<Backend = B>>,
        mrope_positions: Option<&[[u32; 3]]>,
        mrope_delta: i64,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        compute_logits: bool,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        let embedded = if let Some(embedding_norm) = &self.embedding_norm {
            embedding_norm
                .encode(&embedded, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(DecoderError::Backend)?
        } else {
            embedded
        };

        let per_layer_inputs = if let Some(per_layer_embedding) = &self.per_layer_embedding {
            let token_ids = token_ids.expect("per-layer embedding requires token_ids");
            Some(per_layer_embedding.encode(token_ids, &embedded, batch_dim.size(), command_buffer)?)
        } else {
            None
        };

        let transformer_output = self.transformer.encode_multimodal(
            embedded,
            per_layer_inputs.as_ref(),
            mrope_positions,
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            Some(state),
            command_buffer,
        )?;

        if mrope_delta != 0 {
            state.mrope_delta = mrope_delta;
        }

        let logits = if compute_logits && let Some(output_range) = output_range {
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
        let final_hidden = if !compute_logits || hidden_feature_layer_indices.is_some() {
            transformer_output.output
        } else {
            None
        };

        Ok(DecoderEncodeOutput {
            logits,
            hidden_features: transformer_output.hidden_features,
            final_hidden,
        })
    }
}
