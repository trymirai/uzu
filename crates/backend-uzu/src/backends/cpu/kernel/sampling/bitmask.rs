use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Bitmask)]
#[variants(T, f32, f16, bf16)]
pub fn bitmask<T: ArrayElement + Float>(
    #[optional(!in_place)] logits: Option<*const T>,
    bitmask: *const u32,
    processed_logits: *mut T,
    batch_size: u32,
    vocab_size: u32,
    #[specialize] in_place: bool,
) {
    let logits: *const T = match in_place {
        true => processed_logits,
        false => logits.unwrap(),
    };

    let batch_size = batch_size as usize;
    let vocab_size = vocab_size as usize;
    let bitmask_size = vocab_size.div_ceil(32);

    for batch_idx in 0..batch_size {
        for token_idx in 0..vocab_size {
            let global_idx = batch_idx * vocab_size + token_idx;
            let bitmask_idx = batch_idx * bitmask_size + token_idx / 32;
            unsafe {
                let mask = (*bitmask.add(bitmask_idx) >> (token_idx % 32)) & 1;
                *processed_logits.add(global_idx) = match mask != 0 {
                    true => *logits.add(global_idx),
                    false => T::neg_infinity(),
                };
            }
        }
    }
}
