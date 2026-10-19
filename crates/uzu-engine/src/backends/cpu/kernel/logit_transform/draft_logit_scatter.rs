use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::array::ArrayElement;

#[kernel(DraftLogitScatter)]
#[variants(T, f32, bf16)]
pub fn draft_logit_scatter<T: ArrayElement + Float>(
    draft_logits: *const T,
    vocab_slots: *const i32,
    logits: *mut T,
    rows: u32,
    draft_vocab_size: u32,
    vocab_size: u32,
) {
    let (draft_vocab_size, vocab_size) = (draft_vocab_size as usize, vocab_size as usize);
    for row in 0..rows as usize {
        for token in 0..vocab_size {
            let slot = unsafe { *vocab_slots.add(token) };
            let value = if slot >= 0 {
                unsafe { *draft_logits.add(row * draft_vocab_size + slot as usize) }
            } else {
                T::neg_infinity()
            };
            unsafe { *logits.add(row * vocab_size + token) = value };
        }
    }
}
