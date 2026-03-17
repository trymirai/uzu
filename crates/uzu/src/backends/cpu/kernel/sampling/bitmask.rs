use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(Bitmask)]
#[variants(T, f32, f16, bf16)]
pub fn bitmask<T: ArrayElement + Float>(
    #[allow(unused)]
    #[optional(!in_place)]
    logits: Option<*const T>,
    #[allow(unused)] bitmask: *const u32,
    #[allow(unused)] processed_logits: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    let src: *const T = if in_place {
        processed_logits as *const T
    } else {
        logits.unwrap()
    };

    for batch in 0..batch_size {
        let batch_start = (batch * vocab_size) as usize;
        let mask_start = (batch * ((vocab_size + 31) / 32)) as usize;

        for i in 0..vocab_size as usize {
            let word_idx = i / 32;
            let bit_idx = i % 32;
            let allowed = unsafe { (*bitmask.add(mask_start + word_idx) >> bit_idx) & 1 };

            unsafe {
                *processed_logits.add(batch_start + i) = if allowed != 0 {
                    *src.add(batch_start + i)
                } else {
                    T::from(f32::NEG_INFINITY).unwrap()
                };
            };
        }
    }
}
