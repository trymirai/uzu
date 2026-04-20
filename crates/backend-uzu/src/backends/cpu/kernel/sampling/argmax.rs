use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::argmax::ArgmaxPair};

fn argmax_is_better(
    a: &ArgmaxPair,
    b: &ArgmaxPair,
) -> bool {
    a.value > b.value || (a.value == b.value && a.index < b.index)
}

const ARGMAX_INIT: ArgmaxPair = ArgmaxPair {
    value: f32::NEG_INFINITY,
    index: u32::MAX,
};

#[kernel(ArgmaxSingle)]
#[variants(T, f32, f16, bf16)]
pub fn argmax_single<T: ArrayElement + Float>(
    logits_data: *const T,
    final_tokens: *mut u32,
    batch_size: u32,
    vocab_size: u32,
) {
    for batch_idx in 0..batch_size as usize {
        let mut best = ARGMAX_INIT;

        for vocab_idx in 0..vocab_size as usize {
            let global_idx = batch_idx * vocab_size as usize + vocab_idx;
            let value = unsafe { (*logits_data.add(global_idx)).to_f32().unwrap() };
            let candidate = ArgmaxPair {
                value,
                index: vocab_idx as u32,
            };
            if argmax_is_better(&candidate, &best) {
                best = candidate;
            }
        }

        unsafe {
            *final_tokens.add(batch_idx) = best.index;
        }
    }
}

#[kernel(ArgmaxMain)]
#[variants(T, f32, f16, bf16)]
pub fn argmax_main<T: ArrayElement + Float>(
    logits_data: *const T,
    partial_results: *mut ArgmaxPair,
    batch_size: u32,
    vocab_size: u32,
) {
    for batch_idx in 0..batch_size as usize {
        let mut best = ARGMAX_INIT;

        for vocab_idx in 0..vocab_size as usize {
            let global_idx = batch_idx * vocab_size as usize + vocab_idx;
            let value = unsafe { (*logits_data.add(global_idx)).to_f32().unwrap() };
            let candidate = ArgmaxPair {
                value,
                index: vocab_idx as u32,
            };
            if argmax_is_better(&candidate, &best) {
                best = candidate;
            }
        }

        unsafe {
            *partial_results.add(batch_idx) = best;
        }
    }
}

#[kernel(ArgmaxFinal)]
pub fn argmax_final(
    partial_results: *const ArgmaxPair,
    final_tokens: *mut u32,
    batch_size: u32,
    vocab_size: u32,
) {
    let _ = vocab_size;
    for batch_idx in 0..batch_size as usize {
        unsafe {
            let pair = *partial_results.add(batch_idx);
            *final_tokens.add(batch_idx) = pair.index;
        }
    }
}
