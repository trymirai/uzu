use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MinP)]
#[variants(T, f32, f16, bf16)]
pub fn min_p<T: ArrayElement + Float>(
    #[allow(unused)]
    #[optional(!in_place)]
    logits: Option<*const T>,
    #[allow(unused)] processed_logits: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
    #[allow(unused)] min_p: f32,
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

        // Find max logit
        let mut max_logit = f32::NEG_INFINITY;
        for i in 0..vocab_size as usize {
            let v = unsafe { (*src.add(batch_start + i)).to_f32().unwrap() };
            if v > max_logit {
                max_logit = v;
            }
        }

        let threshold = max_logit + min_p.ln();

        for i in 0..vocab_size as usize {
            let v = unsafe { *src.add(batch_start + i) };
            let vf = v.to_f32().unwrap();
            unsafe {
                *processed_logits.add(batch_start + i) = if vf >= threshold {
                    v
                } else {
                    T::from(f32::NEG_INFINITY).unwrap()
                };
            };
        }
    }
}
