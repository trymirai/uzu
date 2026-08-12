use thiserror::Error;

use crate::{
    backends::common::{Backend, Encoder, gpu_types::trie::TrieNode as GpuTrieNode},
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::DecoderError, sampling::PRng},
    engine::language_model::LanguageModel,
    trace::Recorder,
    trie::TrieNode,
    utils::trace::trace_host,
};

/// Attention correctness depends on a hardcoded suffix bound, and the prefill
/// path chunks around it. A trace has to be one uninterrupted pass, so it
/// cannot chunk — see the NOTE in `engine::language_model::stream::stream`.
const MAX_TRACE_TOKENS: usize = 1024;

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
}

impl<B: Backend> LanguageModel<B> {
    /// Runs one forward pass over `token_ids` against a fresh state, capturing
    /// every intermediate activation under lalamo's trace layout.
    ///
    /// This is prefill-shaped on purpose: it mirrors lalamo's tracer, which
    /// evaluates all tokens at once with no cache carried in.
    pub fn record_trace(
        &self,
        token_ids: &[u64],
    ) -> Result<Recorder<B>, RecordTraceError<B>> {
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
        transformer_state.prepare(0, token_count, context).map_err(RecordTraceError::Backend)?;

        let mut encoder = Encoder::<B>::new_with_name(context, Some("trace")).map_err(RecordTraceError::Backend)?;
        encoder.attach_recorder(Recorder::new());

        let mut token_ids_allocation = encoder
            .allocate_constant(token_count * DataType::U32.size_in_bytes())
            .map_err(RecordTraceError::Backend)?;
        token_ids_allocation.copyin(&token_ids.iter().map(|token_id| *token_id as u32).collect::<Box<[u32]>>());

        // lalamo's trace stores ids and positions as i32; uzu feeds the decoder
        // u32, so these two are recorded from the host rather than copied back.
        // A flat trie over a fresh state means positions are exactly `0..n`.
        let host_token_ids = token_ids.iter().map(|token_id| *token_id as i32).collect::<Box<[i32]>>();
        let host_token_positions = (0..token_count as i32).collect::<Box<[i32]>>();
        let token_shape = [1, token_count];
        trace_host!(encoder, "activation_trace.token_ids", &host_token_ids, token_shape, DataType::I32);
        trace_host!(encoder, "activation_trace.token_positions", &host_token_positions, token_shape, DataType::I32);

        let input_trie = TrieNode::flat(0, token_ids, &PRng::new(0));
        let input_flat_trie = input_trie.linearize();
        let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
        let batch_dim = BatchTopology::new(&input_flat_trie_nodes, true);

        // The full output range is what makes the trace comparable: it runs
        // every layer and produces `output_norm` and `logits` for all tokens,
        // rather than just the row that would be sampled.
        self.decoder.encode(
            &token_ids_allocation,
            &batch_dim,
            Some(0..token_count),
            None,
            &mut transformer_state,
            &mut encoder,
        )?;

        // Pooled allocations must be released before the encoder's pool is,
        // otherwise freeing the pool trips over a live range. The captured
        // arrays are unaffected — they are allocated globally.
        drop(token_ids_allocation);

        let mut completed =
            encoder.end_encoding().submit().wait_until_completed().map_err(RecordTraceError::Backend)?;

        Ok(completed.take_recorder().expect("recorder was attached before encoding"))
    }
}
