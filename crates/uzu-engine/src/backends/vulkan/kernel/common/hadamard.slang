import definitions;
import common.workgroup_reduction;
import generated.hadamard_order;

public float hadamard_butterflies(float value, uint lanes) {
  var transformed = value;
  for (uint stride = 1; stride < lanes; stride <<= 1) {
    let partner = shuffle_xor(transformed, stride);
    transformed = (WaveGetLaneIndex() & stride) != 0 ? partner - transformed : partner + transformed;
  }
  return transformed;
}

static const uint HADAMARD_MAX_SIZE = 128;

// Transforms each block of `size` consecutive invocations, a power of two up to HADAMARD_MAX_SIZE that divides THREADS.
// Normalizes by `transform_size`, which is larger when the block is one step of a bigger transform.
public float hadamard_transform<let THREADS : uint>(
  float value,
  uint size = HADAMARD_TRANSFORM_BLOCK_SIZE,
  uint transform_size = HADAMARD_TRANSFORM_BLOCK_SIZE
) {
  static_assert(THREADS % HADAMARD_TRANSFORM_BLOCK_SIZE == 0, "Hadamard transform blocks must tile the workgroup");
  let lanes = WaveGetLaneCount() <= MAX_FAST_SUBGROUP_LANES ? min(WaveGetLaneCount(), size) : 1u;
  var transformed = hadamard_butterflies(value, lanes);
  if (lanes < size) {
    typealias S = WorkgroupScratch<float, THREADS>;
    let invocation = subgroup_invocation();
    S.data[invocation] = transformed;
    GroupMemoryBarrierWithGroupSync();
    let count = size / lanes;
    let first = invocation & ~(size - lanes);
    float column[HADAMARD_MAX_SIZE];
    for (uint j = 0; j < count; j++) {
      column[j] = S.data[first + j * lanes];
    }
    for (uint stride = 1; stride < count; stride <<= 1) {
      for (uint j = 0; j < count; j++) {
        if ((j & stride) == 0) {
          let low = column[j];
          let high = column[j | stride];
          column[j] = high + low;
          column[j | stride] = low - high;
        }
      }
    }
    transformed = column[(invocation % size) / lanes];
    GroupMemoryBarrierWithGroupSync();
  }
  return transformed / sqrt(float(transform_size));
}
