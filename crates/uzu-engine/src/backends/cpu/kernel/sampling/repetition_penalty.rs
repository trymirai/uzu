use half::bf16;
use num_traits::{Float, NumCast};
use uzu_engine_macros::kernel;

use crate::{
    backends::common::gpu_types::{ring::RingParams, trie::TrieNode},
    data_type::DataTypeRepr,
};

#[kernel(RepetitionPenalty)]
#[variants(T, f32, bf16)]
pub fn repetition_penalty<T: DataTypeRepr + Float + NumCast>(
    original_logits: *const T,
    logits_copy: *mut T,
    context_ring: *const u32,
    token_ids: *const u32,
    #[optional(is_trie)] trie: Option<*const TrieNode>,
    repetition_penalty: f32,
    suffix_repetition_length: u32,
    vocab_size: u32,
    sampling_start: u32,
    sampling_length: u32,
    #[specialize] is_trie: bool,
) {
    fn apply(
        original_logits: *const (impl DataTypeRepr + Float),
        logits_copy: *mut (impl DataTypeRepr + Float),
        vocab_size: usize,
        sample_index: usize,
        token_id: usize,
        repetition_penalty: f32,
    ) {
        let offset = sample_index * vocab_size + token_id;
        let logit = unsafe { (*original_logits.add(offset)).to_f32().unwrap() };
        let penalized = if logit > 0.0 {
            logit / repetition_penalty
        } else {
            logit * repetition_penalty
        };
        unsafe {
            *logits_copy.add(offset) = NumCast::from(penalized).unwrap();
        }
    }

    assert_eq!(trie.is_some(), is_trie);
    let ring = unsafe { &*context_ring.cast::<RingParams>() };
    let ring_tokens = unsafe { context_ring.add(2) };
    let vocab_size = vocab_size as usize;

    for sample_index in 0..sampling_length {
        let batch_index = sampling_start + sample_index;
        let batch_prefix_length =
            trie.map_or(batch_index, |trie| unsafe { (*trie.add(batch_index as usize)).height }) + 1;
        let total_length = ring.ring_length + batch_prefix_length;
        let window_length = total_length.min(suffix_repetition_length);
        let skipped_length = total_length - window_length;

        for window_index in 0..window_length {
            let source_index = skipped_length + window_index;
            let token_id = if source_index < ring.ring_length {
                let slot = (ring.ring_offset + source_index) % suffix_repetition_length;
                (unsafe { *ring_tokens.add(slot as usize) }) as usize
            } else {
                let mut token_index = source_index - ring.ring_length;
                if let Some(trie) = trie {
                    let height = token_index;
                    token_index = batch_index;
                    while unsafe { (*trie.add(token_index as usize)).height } != height {
                        token_index -= 1;
                    }
                }
                (unsafe { *token_ids.add(token_index as usize) }) as usize
            };
            apply(original_logits, logits_copy, vocab_size, sample_index as usize, token_id, repetition_penalty);
        }
    }
}
