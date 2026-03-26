use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MinP)]
#[variants(T, f32, f16, bf16)]
pub fn min_p<T: ArrayElement + Float>(
    #[optional(!in_place)] logits: Option<*const T>,
    processed_logits: *mut T,
    batch_size: u32,
    vocab_size: u32,
    min_p: f32,
    #[specialize] in_place: bool,
) {
    let logits = match in_place {
        true => processed_logits,
        false => logits.unwrap(),
    };

    let batch_size = batch_size as usize;
    let vocab_size = vocab_size as usize;

    for batch_idx in 0..batch_size {
        // find maximum logit
        let mut max_logit = f32::NEG_INFINITY;
        for i in 0..vocab_size {
            let logit = unsafe { *logits.add(batch_idx * vocab_size + i) }.to_f32().unwrap();
            max_logit = max_logit.max(logit);
        }

        // then the threshold is just max_logit + log(min_p), mask everything strictly below it
        let threshold: T = T::from(max_logit + min_p.ln()).unwrap();
        for i in 0..vocab_size {
            let position = batch_idx * vocab_size + i;
            unsafe {
                let logit = *logits.add(position);
                *processed_logits.add(position) = match logit >= threshold {
                    true => logit,
                    false => T::neg_infinity(),
                }
            }
        }
    }
}
