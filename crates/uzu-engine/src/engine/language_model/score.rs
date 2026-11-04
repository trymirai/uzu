use half::bf16;
use thiserror::Error;

use super::LanguageModel;
use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode,
    },
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::DecoderError},
};

#[derive(Debug)]
pub struct TokenScores {
    pub log_probabilities: Box<[f64]>,
    pub label_mass: f64,
    pub prefill_duration: std::time::Duration,
}

#[derive(Debug, Error)]
pub enum TokenScoreError<B: Backend> {
    #[error("prompt must not be empty")]
    EmptyPrompt,
    #[error("labels must not be empty")]
    EmptyLabels,
    #[error("label token IDs must be distinct")]
    DuplicateLabel,
    #[error("token ID is outside the vocabulary")]
    InvalidToken,
    #[error("prompt exceeds the context limit")]
    ContextOverflow,
    #[error("model returned invalid logits")]
    InvalidModelOutput,
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("GPU completion failed: {0}")]
    Completion(String),
    #[error("decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
}

impl<B: Backend> LanguageModel<B> {
    /// Scores continuations at the answer boundary without sampling or creating chat state.
    /// Native scoring blocks through GPU completion; use a blocking worker in async servers.
    pub async fn score_token_ids(
        &self,
        prompt: &[u64],
        labels: &[u64],
        max_context_length: Option<u32>,
    ) -> Result<TokenScores, TokenScoreError<B>> {
        let vocab_size = self.decoder.embedding().vocab_size() as usize;
        if prompt.is_empty() {
            return Err(TokenScoreError::EmptyPrompt);
        }
        if labels.is_empty() {
            return Err(TokenScoreError::EmptyLabels);
        }
        if prompt.iter().chain(labels).any(|&id| id >= vocab_size as u64) {
            return Err(TokenScoreError::InvalidToken);
        }
        if labels.iter().enumerate().any(|(index, id)| labels[..index].contains(id)) {
            return Err(TokenScoreError::DuplicateLabel);
        }
        let context_limit = match (self.max_context_length(), max_context_length) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if prompt.len() > u32::MAX as usize || context_limit.is_some_and(|limit| prompt.len() > limit as usize) {
            return Err(TokenScoreError::ContextOverflow);
        }
        let prefill_started = std::time::Instant::now();
        let context = &self.engine.context;
        let mut state = self.decoder.create_empty_state(context_limit, context).map_err(TokenScoreError::Backend)?;
        let batches = prompt.len().div_ceil(1024);
        state
            .prepare(((batches - 1) * 1024) as u32, prompt.len().min(1024) as u32)
            .map_err(TokenScoreError::Backend)?;
        let mut output_buffer =
            context.create_buffer_gpu(vocab_size * DataType::BF16.size_in_bytes()).map_err(TokenScoreError::Backend)?;
        let mut commands =
            context.create_command_buffer(Some("score prefill"), None).map_err(TokenScoreError::Backend)?;
        let split_last = self.decoder.prefill_cache_skips_trailing_layers();
        for (batch, chunk) in prompt.chunks(1024).enumerate() {
            let final_batch = batch + 1 == batches;
            let parts = if final_batch && split_last && chunk.len() > 1 {
                let (prefix, last) = chunk.split_at(chunk.len() - 1);
                [Some((prefix, false)), Some((last, true))]
            } else {
                [Some((chunk, final_batch)), None]
            };
            for (part, readout) in parts.into_iter().flatten() {
                let count = part.len() as u32;
                let nodes: Box<[_]> = (0..count)
                    .map(|index| TrieNode {
                        trie_start: index,
                        trie_end: count,
                        height: index,
                    })
                    .collect();
                let topology = BatchTopology::new(&nodes, true);
                let ids = commands
                    .allocate_constant_from_slice(&part.iter().map(|id| *id as u32).collect::<Box<[_]>>())
                    .map_err(TokenScoreError::Backend)?;
                let output = self.decoder.encode(
                    &ids,
                    &topology,
                    readout.then(|| (count - 1..count).into()),
                    None,
                    &mut state,
                    &mut commands,
                )?;
                if let Some(logits) = output.logits.as_ref() {
                    commands.encode_copy(logits, &mut output_buffer);
                }
                state
                    .encode_accept(&(0..count).collect::<Box<[_]>>(), &mut commands)
                    .map_err(TokenScoreError::Backend)?;
                drop(output);
                drop(ids);
            }
        }
        let (logits, prefill_duration) = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    futures::executor::block_on(async {
                        commands.end_encoding().submit().wait_until_completed().await.map_err(|e| e.to_string())?;
                        let prefill_duration = prefill_started.elapsed();
                        let logits = context.copyout::<bf16>(&output_buffer).await.map_err(|e| e.to_string())?;
                        Ok::<_, String>((logits, prefill_duration))
                    })
                })
                .join()
                .map_err(|_| TokenScoreError::Completion("completion worker failed".into()))
        })?
        .map_err(TokenScoreError::Completion)?;
        if logits.len() != vocab_size {
            return Err(TokenScoreError::InvalidModelOutput);
        }
        let mut scores =
            scores_from_logits::<B>(&logits.iter().map(|value| value.to_f32()).collect::<Vec<_>>(), labels)?;
        scores.prefill_duration = prefill_duration;
        Ok(scores)
    }
}

fn scores_from_logits<B: Backend>(
    logits: &[f32],
    labels: &[u64],
) -> Result<TokenScores, TokenScoreError<B>> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(TokenScoreError::InvalidModelOutput);
    }
    let max = logits.iter().copied().map(f64::from).fold(f64::NEG_INFINITY, f64::max);
    let normalizer = logits.iter().map(|&value| (f64::from(value) - max).exp()).sum::<f64>().ln();
    let log_probabilities: Box<[_]> =
        labels.iter().map(|&id| (f64::from(logits[id as usize]) - max) - normalizer).collect();
    let label_max = log_probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_mass = label_max + log_probabilities.iter().map(|value| (value - label_max).exp()).sum::<f64>().ln();
    let label_mass = log_mass.exp().min(1.0);
    if !normalizer.is_finite() || log_probabilities.iter().any(|value| !value.is_finite()) || !label_mass.is_finite() {
        return Err(TokenScoreError::InvalidModelOutput);
    }
    Ok(TokenScores {
        log_probabilities,
        label_mass,
        prefill_duration: std::time::Duration::ZERO,
    })
}

#[cfg(test)]
#[path = "../../../unit/engine/language_model/score_test.rs"]
mod tests;
