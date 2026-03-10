use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TopK)]
#[variants(T, f32, f16, bf16)]
pub fn top_k<T: ArrayElement + Float>(
    #[optional(!in_place)] logits: Option<*const T>,
    processed_logits: *mut T,
    batch_size: u32,
    vocab_size: u32,
    top_k: u32,
    #[specialize] in_place: bool,
) {
    if top_k == 0 || top_k > vocab_size {
        return;
    }

    let logits: *const T = match in_place {
        true => processed_logits,
        false => logits.unwrap(),
    };

    let vocab_size = vocab_size as usize;
    let top_k = top_k as usize;

    for batch_idx in 0..batch_size as usize {
        let batch_start = batch_idx * vocab_size;
        let mut sorted_logits = unsafe { std::slice::from_raw_parts(logits.add(batch_start), vocab_size) }.to_vec();
        sorted_logits.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let t_threshold = sorted_logits[top_k - 1];

        // Mask everything below the threshold
        for i in 0..vocab_size {
            let logit_value = unsafe { *logits.add(batch_start + i) };
            unsafe {
                *processed_logits.add(batch_start + i) = if logit_value >= t_threshold {
                    logit_value
                } else {
                    T::from(f32::NEG_INFINITY).unwrap()
                };
            }
        }
    }
}
