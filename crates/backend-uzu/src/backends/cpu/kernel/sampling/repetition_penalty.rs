use half::bf16;
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::ring::RingParams};

#[kernel(RepetitionPenalty)]
#[variants(T, f32, bf16)]
pub fn repetition_penalty<T: ArrayElement + Float + NumCast>(
    original_logits: *const T,
    logits_copy: *mut T,
    context_ring: *const u32,
    token_ids: *const u64,
    repetition_penalty: f32,
    suffix_repetition_length: u32,
) {
    fn apply<T: ArrayElement + Float + NumCast>(
        original_logits: *const T,
        logits_copy: *mut T,
        token_id: usize,
        repetition_penalty: f32,
    ) {
        let logit = unsafe { (*original_logits.add(token_id)).to_f32().unwrap() };
        let penalized = if logit > 0.0 {
            logit / repetition_penalty
        } else {
            logit * repetition_penalty
        };
        unsafe {
            *logits_copy.add(token_id) = NumCast::from(penalized).unwrap();
        }
    }

    let ring = unsafe { &*context_ring.cast::<RingParams>() };
    let ring_tokens = unsafe { context_ring.add(2) };

    apply(original_logits, logits_copy, unsafe { *token_ids } as usize, repetition_penalty);

    for ring_index in 0..suffix_repetition_length {
        if ring_index >= ring.ring_length {
            continue;
        }
        let slot = (ring.ring_offset + ring_index) % suffix_repetition_length;
        apply(original_logits, logits_copy, unsafe { *ring_tokens.add(slot as usize) } as usize, repetition_penalty)
    }
}
