use std::cmp::Ordering;

use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    language_model::gumbel::{gumbel_float, revidx},
};

// NOTE: top_k + top_p combination is not exactly matching lalamo ("parallel" here, should be top-k then top-p)
#[kernel(UnifiedSampling)]
#[variants(T, f32, bf16)]
pub fn unified_sampling<T: ArrayElement + Float>(
    logits: *const T,
    output: *mut u32,
    #[optional(is_stochastic)] seeds: Option<*const u64>,
    #[optional(has_bitmask)] bitmask: Option<*const u32>,
    #[optional(has_temperature)] temperature: Option<f32>,
    #[optional(has_top_k)] top_k: Option<u32>,
    #[optional(has_top_p)] top_p: Option<f32>,
    #[optional(has_min_p)] min_p: Option<f32>,
    vocab_size: u32,
    batch_size: u32,
    #[specialize] is_stochastic: bool,
    #[specialize] has_bitmask: bool,
    #[specialize] has_temperature: bool,
    #[specialize] temperature_after_filters: bool,
    #[specialize] has_top_k: bool,
    #[specialize] has_top_p: bool,
    #[specialize] has_min_p: bool,
) {
    for batch_idx in 0..batch_size {
        let mut logits = unsafe {
            std::slice::from_raw_parts(logits.wrapping_add((vocab_size * batch_idx) as usize), vocab_size as usize)
        }
        .iter()
        .map(|logit| logit.to_f32().unwrap())
        .collect::<Vec<f32>>();

        if has_bitmask {
            let bitmask = unsafe {
                std::slice::from_raw_parts(
                    bitmask.unwrap().wrapping_add((vocab_size.div_ceil(u32::BITS) * batch_idx) as usize),
                    vocab_size.div_ceil(u32::BITS) as usize,
                )
            };
            for (logit_index, logit) in logits.iter_mut().enumerate() {
                if bitmask[logit_index / (u32::BITS as usize)] & (1 << (logit_index % (u32::BITS as usize))) == 0 {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        if has_temperature && !temperature_after_filters {
            let recip_temperature = 1.0 / temperature.unwrap();
            for logit in logits.iter_mut() {
                *logit *= recip_temperature;
            }
        }

        if has_top_k || has_top_p || has_min_p {
            let mut sorted_logits = logits.iter().copied().enumerate().collect::<Vec<_>>();
            sorted_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));

            let logits_max = sorted_logits[0].1;
            let logits_norm = sorted_logits.iter().map(|logit| (logit.1 - logits_max).exp()).sum::<f32>();

            logits.fill(f32::NEG_INFINITY);
            let mut top_p_mass = 0.0;
            for (top_k_num, (index, logit)) in sorted_logits.into_iter().enumerate() {
                if (has_top_k && top_k_num as u32 >= top_k.unwrap())
                    || (has_top_p && top_p_mass >= top_p.unwrap())
                    || (has_min_p && logit < logits_max + min_p.unwrap().ln())
                {
                    break;
                }
                logits[index] = logit;
                top_p_mass += (logit - logits_max).exp() / logits_norm;
            }
        }

        if has_temperature && temperature_after_filters {
            let recip_temperature = 1.0 / temperature.unwrap();
            for logit in logits.iter_mut() {
                *logit *= recip_temperature;
            }
        }

        if is_stochastic {
            let seed = unsafe { *seeds.unwrap().wrapping_add(batch_idx as usize) };
            for (logit_index, logit) in logits.iter_mut().enumerate() {
                *logit += gumbel_float(seed, revidx(logit_index as u32, vocab_size));
            }
        }

        let argmax = logits
            .into_iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(b.0.cmp(&a.0)))
            .unwrap()
            .0;

        unsafe { *output.wrapping_add(batch_idx as usize) = argmax as u32 }
    }
}
