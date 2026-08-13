use std::{collections::HashMap, path::Path};

use thiserror::Error;

use crate::{
    backends::common::{Backend, Encoder, gpu_types::trie::TrieNode as GpuTrieNode},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology, decoder::DecoderError, mixer::attention::ATTENTION_SUFFIX_CAPACITY,
        sampling::PRng,
    },
    engine::language_model::LanguageModel,
    trace::{Array, DecoderTap, DecoderTapRequest},
    trie::TrieNode,
};

// A trace is a single pass, so it cannot chunk around the suffix bound like prefill does.
const MAX_TRACE_TOKENS: usize = ATTENTION_SUFFIX_CAPACITY;

#[derive(Debug, Error)]
pub enum RecordTraceError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("Trace input is empty")]
    EmptyInput,
    #[error("Trace input has {token_count} tokens, more than the {MAX_TRACE_TOKENS} a single pass supports")]
    TooManyTokens {
        token_count: usize,
    },
    #[error("Trace error: {0}")]
    Trace(#[from] crate::trace::Error),
}

impl<B: Backend> LanguageModel<B> {
    pub fn tap(&self) -> &DecoderTap<B> {
        &self.tap
    }

    pub fn write_trace(
        &self,
        output_path: &Path,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<(), crate::trace::Error> {
        if let Some(parent) = output_path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        self.tap.write(output_path, metadata)
    }

    /// One prefill-shaped pass over `token_ids` against a fresh state.
    pub fn record_trace(
        &mut self,
        token_ids: &[u64],
        request: &DecoderTapRequest,
    ) -> Result<&DecoderTap<B>, RecordTraceError<B>> {
        let token_count = token_ids.len();
        if token_count == 0 {
            return Err(RecordTraceError::EmptyInput);
        }
        if token_count > MAX_TRACE_TOKENS {
            return Err(RecordTraceError::TooManyTokens {
                token_count,
            });
        }

        let context = &self.engine.context;
        let mut transformer_state =
            self.decoder.create_empty_state(Some(token_count), context).map_err(RecordTraceError::Backend)?;
        transformer_state
            .prepare(transformer_state.context_length(), token_count, context)
            .map_err(RecordTraceError::Backend)?;

        let mut encoder = Encoder::<B>::new_with_name(context, Some("trace")).map_err(RecordTraceError::Backend)?;

        let mut token_ids_allocation = encoder
            .allocate_constant(token_count * DataType::U32.size_in_bytes())
            .map_err(RecordTraceError::Backend)?;
        token_ids_allocation.copyin(&token_ids.iter().map(|token_id| *token_id as u32).collect::<Box<[u32]>>());

        let input_trie = TrieNode::flat(0, token_ids, &PRng::new(0));
        let input_flat_trie = input_trie.linearize();
        let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
        let batch_dim = BatchTopology::new(&input_flat_trie_nodes, true);

        // Full output range: every layer, all tokens. Taking `.tap` off the temporary
        // drops the rest, whose pooled logits must not outlive the encoder's pool.
        let mut tap = self
            .decoder
            .encode(
                &token_ids_allocation,
                &batch_dim,
                Some(0..token_count),
                None,
                &mut transformer_state,
                Some(request),
                &mut encoder,
            )?
            .tap;

        // Flat trie over a fresh state gives positions 0..n.
        if let Some(transformer_tap) = &mut tap.transformer {
            let shape = [1, token_count];
            let host_token_ids = token_ids.iter().map(|token_id| *token_id as i32).collect::<Box<[i32]>>();
            let host_token_positions = (0..token_count as i32).collect::<Box<[i32]>>();
            transformer_tap.token_ids = Some(
                Array::capture_slice(&encoder, &host_token_ids, &shape, DataType::I32)
                    .map_err(RecordTraceError::Backend)?,
            );
            transformer_tap.token_positions = Some(
                Array::capture_slice(&encoder, &host_token_positions, &shape, DataType::I32)
                    .map_err(RecordTraceError::Backend)?,
            );
        }

        // Pooled allocations must be released before the pool is.
        drop(token_ids_allocation);

        encoder.end_encoding().submit().wait_until_completed().map_err(RecordTraceError::Backend)?;

        self.tap = tap;

        Ok(&self.tap)
    }
}
