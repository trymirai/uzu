use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    ArrayElement,
    language_model::gumbel::{gumbel_float, revidx},
};

#[kernel(Gumbel)]
#[variants(T, f32, f16, bf16)]
pub fn gumbel<T: ArrayElement + Float>(
    #[optional(!in_place)] logits: Option<*const T>,
    batch_seeds: *const u64,
    batch_seeds_offset: u32,
    processed_logits: *mut T,
    batch_size: u32,
    vocab_size: u32,
    #[specialize] in_place: bool,
) {
    let logits: *const T = match in_place {
        true => processed_logits,
        false => logits.unwrap(),
    };

    for batch_idx in 0..batch_size as usize {
        let batch_start = batch_idx * vocab_size as usize;
        let rng_seed = unsafe { *batch_seeds.add(batch_seeds_offset as usize + batch_idx) };
        for vocab_idx in 0..vocab_size as usize {
            let global_idx = batch_start + vocab_idx;
            unsafe {
                let logit = (*logits.add(global_idx)).to_f32().unwrap();
                let gumbel_noise = gumbel_float(rng_seed, revidx(vocab_idx as u32));
                *processed_logits.add(global_idx) = T::from(logit + gumbel_noise).unwrap();
            }
        }
    }
}
