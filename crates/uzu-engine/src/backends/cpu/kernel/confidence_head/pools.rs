use uzu_engine_macros::kernel;

use super::head::{LOG_PROB_FLOOR, log_sum};
use crate::{
    backends::common::gpu_types::confidence_head::ConfidenceHeadParams,
    encodable_block::sampling::{gumbel_float, revidx},
};

#[kernel(ConfidenceHeadPools)]
pub fn confidence_head_pools(
    candidate_logits: *const f32,
    candidate_ids: *const u32,
    depth_seeds: *const u64,
    pool_features: *mut f32,
    params: ConfidenceHeadParams,
    depths: u32,
) {
    let candidates = params.candidates as usize;
    for depth in 0..depths as usize {
        let logits = unsafe { std::slice::from_raw_parts(candidate_logits.add(depth * candidates), candidates) };
        let log_sum = log_sum(logits);
        let seed = unsafe { *depth_seeds.add(depth) };
        for (index, logit) in logits.iter().enumerate() {
            let rank = logits.iter().filter(|other| *other > logit).count() as f32;
            let token = unsafe { *candidate_ids.add(depth * candidates + index) };
            let features = unsafe { pool_features.add(depth * 3 * candidates) };
            unsafe {
                *features.add(index) = (logit - log_sum).max(LOG_PROB_FLOOR);
                *features.add(candidates + index) = gumbel_float(seed, revidx(token, params.vocab_size));
                *features.add(2 * candidates + index) = rank.ln_1p();
            }
        }
    }
}
