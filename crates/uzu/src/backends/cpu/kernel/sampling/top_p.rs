use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TopP)]
#[variants(T, f32, f16, bf16)]
pub fn top_p<T: ArrayElement + Float>(
    #[optional(!in_place)] logits: Option<*const T>,
    processed_logits: *mut T,
    batch_size: u32,
    vocab_size: u32,
    top_p: f32,
    #[specialize] in_place: bool,
) {
    let logits: *const T = match in_place {
        true => processed_logits,
        false => logits.unwrap(),
    };

    let vocab_size = vocab_size as usize;

    for batch_idx in 0..batch_size as usize {
        let batch_start = batch_idx * vocab_size;

        // Sort logits in descending order
        let mut sorted_logits = unsafe { std::slice::from_raw_parts(logits.add(batch_start), vocab_size) }.to_vec();
        sorted_logits.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let max_logit: f32 = sorted_logits[0].to_f32().unwrap();

        // Compute softmax probabilities and find threshold via cumulative sum
        let total_sum: f32 = sorted_logits
            .iter()
            .map(|&v| v.to_f32().unwrap())
            .filter(|&v| v > f32::NEG_INFINITY)
            .map(|v| (v - max_logit).exp())
            .sum();

        let mut cumulative: f32 = 0.0;
        let mut threshold: f32 = sorted_logits[0].to_f32().unwrap();
        for &logit in &sorted_logits {
            let logit_f32 = logit.to_f32().unwrap();
            if logit_f32 == f32::NEG_INFINITY {
                break;
            }
            threshold = logit_f32;
            cumulative += (logit_f32 - max_logit).exp() / total_sum;
            if cumulative >= top_p {
                break;
            }
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
