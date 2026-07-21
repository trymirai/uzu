#include <metal_stdlib>
#include "../common/dsl.h"
#include "weaver_frontier.h"

using namespace metal;

PUBLIC KERNEL(WeaverCandidateGather)(
    const device uint* pool_ids,
    const device float* pool_scores,
    const device uint* round_metadata,
    device uint* candidate_ids,
    device float* candidate_scores,
    constant uint& rows,
    constant uint& pool_rows,
    constant uint& pool_size,
    const uint position AXIS(rows* pool_size, 256)
) {
  if (pool_size == 0 || pool_rows == 0) {
    return;
  }

  const uint row = position / pool_size, index = position % pool_size;
  const uint source = min(round_metadata[WEAVER_METADATA_LANE_DEPTH * rows + row], pool_rows - 1u);

  candidate_ids[row * pool_size + index] = pool_ids[source * pool_size + index];
  candidate_scores[row * pool_size + index] = pool_scores[source * pool_size + index];
}
