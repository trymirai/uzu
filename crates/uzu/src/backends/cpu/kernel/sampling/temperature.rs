use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Temperature)]
#[variants(T, f32, f16, bf16)]
pub fn temperature<T: ArrayElement + Float>(
    #[optional(!in_place)] logits: Option<*const T>,
    processed_logits: *mut T,
    batch_size: u32,
    vocab_size: u32,
    temperature: f32,
    #[specialize] in_place: bool,
) {
    let logits: *const T = match in_place {
        true => processed_logits,
        false => logits.unwrap(),
    };

    for batch_idx in 0..batch_size as usize {
        for vocab_idx in 0..vocab_size as usize {
            let global_idx = batch_idx * vocab_size as usize + vocab_idx;
            unsafe {
                let value: f32 = (*logits.add(global_idx)).to_f32().unwrap() / temperature;
                *processed_logits.add(global_idx) = T::from(value).unwrap();
            }
        }
    }
}
