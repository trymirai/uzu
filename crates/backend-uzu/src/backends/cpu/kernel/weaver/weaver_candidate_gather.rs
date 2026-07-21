use proc_macros::kernel;

use crate::backends::common::kernel::weaver::METADATA_LANE_DEPTH;

#[kernel(WeaverCandidateGather)]
pub fn weaver_candidate_gather(
    pool_ids: *const u32,
    pool_scores: *const f32,
    round_metadata: *const u32,
    candidate_ids: *mut u32,
    candidate_scores: *mut f32,
    rows: u32,
    pool_rows: u32,
    pool_size: u32,
) {
    if pool_size == 0 || pool_rows == 0 {
        return;
    }
    unsafe {
        let (rows, size) = (rows as usize, pool_size as usize);
        for dst in 0..rows * size {
            let row = dst / size;
            let src =
                (*round_metadata.add(METADATA_LANE_DEPTH * rows + row)).min(pool_rows - 1) as usize * size + dst % size;
            *candidate_ids.add(dst) = *pool_ids.add(src);
            *candidate_scores.add(dst) = *pool_scores.add(src);
        }
    }
}
