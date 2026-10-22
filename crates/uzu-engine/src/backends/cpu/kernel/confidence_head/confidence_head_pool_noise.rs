use uzu_engine_macros::kernel;

use crate::encodable_block::sampling::{gumbel_float, revidx};

/// The target sampler's Gumbel noise on every candidate of every pool row: row d draws at the seed of depth d,
/// the seed the target samples the children of a depth-d node with.
#[kernel(ConfidenceHeadPoolNoise)]
pub fn confidence_head_pool_noise(
    candidate_ids: *const u32,
    depth_seeds: *const u64,
    noise: *mut f32,
    count: u32,
    candidates: u32,
    vocab_size: u32,
) {
    for position in 0..count as usize {
        let row = position / candidates as usize;
        unsafe {
            let seed = *depth_seeds.add(row);
            *noise.add(position) = gumbel_float(seed, revidx(*candidate_ids.add(position), vocab_size));
        }
    }
}
