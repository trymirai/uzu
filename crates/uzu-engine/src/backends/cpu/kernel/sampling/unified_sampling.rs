use std::cmp::Ordering;

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    array::ArrayElement,
    encodable_block::sampling::{gumbel_float, revidx},
};

// Best first: numeric descending with signed zeros equal, NaN after every number, lowest index on ties.
fn logit_order(
    a: &(usize, f32),
    b: &(usize, f32),
) -> Ordering {
    b.1.partial_cmp(&a.1).unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan())).then(a.0.cmp(&b.0))
}

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

        if has_temperature {
            let recip_temperature = 1.0 / temperature.unwrap();
            for logit in logits.iter_mut() {
                *logit *= recip_temperature;
            }
        }

        if has_top_k || has_top_p || has_min_p {
            let mut sorted_logits = logits.iter().copied().enumerate().collect::<Vec<_>>();
            sorted_logits.sort_by(logit_order);

            let logits_max = sorted_logits[0].1;
            let logits_norm = sorted_logits.iter().map(|logit| (logit.1 - logits_max).exp()).sum::<f32>();

            // NaN ranks after every kept logit, including -inf
            logits.fill(f32::NAN);
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

        if is_stochastic {
            let seed = unsafe { *seeds.unwrap().wrapping_add(batch_idx as usize) };
            for (logit_index, logit) in logits.iter_mut().enumerate() {
                *logit += gumbel_float(seed, revidx(logit_index as u32, vocab_size));
            }
        }

        let argmax = logits.into_iter().enumerate().min_by(logit_order).unwrap().0;

        unsafe { *output.wrapping_add(batch_idx as usize) = argmax as u32 }
    }
}

#[cfg(test)]
#[uzu_engine_macros::uzu_test]
fn logit_order_is_a_total_order() {
    let old_sort =
        |a: &(usize, f32), b: &(usize, f32)| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0));
    let old_argmax =
        |a: &(usize, f32), b: &(usize, f32)| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(b.0.cmp(&a.0));
    // The old sort comparator is cyclic once a NaN is present: (0, 1) < (1, NaN) < (2, 2) < (0, 1).
    let (one, nan, two) = ((0, 1.0), (1, f32::NAN), (2, 2.0));
    assert_eq!([old_sort(&one, &nan), old_sort(&nan, &two), old_sort(&two, &one)], [Ordering::Less; 3]);
    assert!(logit_order(&two, &one).is_lt() && logit_order(&one, &nan).is_lt());

    let values = [f32::NAN, f32::NEG_INFINITY, f32::MIN, -1.0, -0.0, 0.0, 1e-45, 1.0, f32::MAX, f32::INFINITY];
    let elements = values.into_iter().flat_map(|value| (0..3).map(move |index| (index, value))).collect::<Vec<_>>();
    for a in &elements {
        for b in &elements {
            let order = logit_order(a, b);
            assert_eq!(order, logit_order(b, a).reverse());
            assert_eq!(order.is_eq(), a.0 == b.0 && (a.1 == b.1 || (a.1.is_nan() && b.1.is_nan())));
            match (a.1.is_nan(), b.1.is_nan()) {
                (false, false) => assert_eq!([old_sort(a, b), old_argmax(a, b).reverse()], [order; 2]),
                (false, true) => assert_eq!(order, Ordering::Less),
                (true, false) => assert_eq!(order, Ordering::Greater),
                (true, true) => assert_eq!(order, a.0.cmp(&b.0)),
            }
            for c in &elements {
                if order.is_le() && logit_order(b, c).is_le() {
                    assert!(logit_order(a, c).is_le());
                }
            }
        }
    }
}
