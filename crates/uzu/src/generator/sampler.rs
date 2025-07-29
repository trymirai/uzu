use ndarray::{ArrayView, IxDyn};

#[cfg(target_arch = "aarch64")]
use crate::backends::cpu::argmax::neon_optimized_argmax;
use crate::{ArrayElement, backends::cpu::argmax::simple_argmax};

pub trait LogitsSampler {
    fn sample<T: ArrayElement>(
        &self,
        logits: ArrayView<T, IxDyn>,
    ) -> Vec<u64>;
}

pub struct ArgmaxSampler {}

impl LogitsSampler for ArgmaxSampler {
    fn sample<T: ArrayElement>(
        &self,
        logits: ArrayView<T, IxDyn>,
    ) -> Vec<u64> {
        let mut result = Vec::<u64>::with_capacity(logits.shape()[0]);

        for row in logits.rows() {
            let max_index = if let Some(slice) = row.as_slice_memory_order() {
                #[cfg(target_arch = "aarch64")]
                {
                    if slice.len() >= 16 {
                        neon_optimized_argmax(slice)
                    } else {
                        simple_argmax(slice)
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    simple_argmax(slice)
                }
            } else {
                let data: Vec<T> = row.iter().cloned().collect();
                simple_argmax(&data)
            };

            result.push(max_index as u64);
        }

        result
    }
}
