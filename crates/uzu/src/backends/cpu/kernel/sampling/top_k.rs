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
    let logits: *const T = match in_place {
        true => processed_logits,
        false => logits.unwrap(),
    };

    let vocab_size = vocab_size as usize;
    let top_k = top_k as usize;

    for batch_idx in 0..batch_size as usize {
        let batch_start = batch_idx * vocab_size;

        // Find min and max logit for binary search
        let mut max_logit = f32::NEG_INFINITY;
        let mut min_logit = f32::INFINITY;
        for i in 0..vocab_size {
            let logit_value = unsafe { (*logits.add(batch_start + i)).to_f32().unwrap() };
            max_logit = max_logit.max(logit_value);
            if logit_value > f32::NEG_INFINITY {
                min_logit = min_logit.min(logit_value);
            }
        }

        // Binary search for the threshold
        let mut low = min_logit;
        let mut high = max_logit;
        let mut threshold = (min_logit + max_logit) / 2.0;
        for _ in 0..16 {
            let mut num_above_threshold: usize = 0;
            for i in 0..vocab_size {
                let logit_value = unsafe { (*logits.add(batch_start + i)).to_f32().unwrap() };
                if logit_value >= threshold {
                    num_above_threshold += 1;
                }
            }

            if num_above_threshold == top_k {
                break;
            } else if num_above_threshold > top_k {
                low = threshold;
            } else {
                high = threshold;
            }
            threshold = (low + high) / 2.0;
        }

        let t_threshold = T::from(threshold).unwrap();

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
