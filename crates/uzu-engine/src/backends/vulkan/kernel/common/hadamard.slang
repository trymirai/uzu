import definitions;
import common.workgroup_reduction;
import generated.hadamard_order;

// A random Hadamard factor applied to a value.
public float hadamard_factor(float value, int factor) {
  return value * float(factor);
}

// The unnormalized Hadamard butterflies over each aligned run of `lanes` lanes, a power of two no wider than the subgroup.
public float hadamard_butterflies(float value, uint lanes) {
  var transformed = value;
  for (uint stride = 1; stride < lanes; stride <<= 1) {
    let partner = shuffle_xor(transformed, stride);
    transformed = (WaveGetLaneIndex() & stride) != 0 ? partner - transformed : partner + transformed;
  }
  return transformed;
}

// hadamard_transform where a block is one aligned run of a subgroup's lanes, for subgroups of at least
// HADAMARD_TRANSFORM_BLOCK_SIZE lanes; equal to it there.
public float subgroup_hadamard_transform(float value) {
  return hadamard_butterflies(value, HADAMARD_TRANSFORM_BLOCK_SIZE) / sqrt(float(HADAMARD_TRANSFORM_BLOCK_SIZE));
}

// The normalized Hadamard transform of blocks of HADAMARD_TRANSFORM_BLOCK_SIZE invocations in subgroup_invocation
// order: shuffle_xor butterflies below the subgroup width, the wider strides in one shared stage. Subgroups wider than
// HADAMARD_SHUFFLE_LANES take every stride in the shared stage: their shuffles move data across far more lanes than a
// block, which measured slower than shared memory on 128-lane subgroups.
static const uint HADAMARD_SHUFFLE_LANES = 64;
public float hadamard_transform<let THREADS : uint>(float value) {
  static_assert(THREADS % HADAMARD_TRANSFORM_BLOCK_SIZE == 0, "Hadamard transform blocks must tile the workgroup");
  let lanes = WaveGetLaneCount() <= HADAMARD_SHUFFLE_LANES ? min(WaveGetLaneCount(), HADAMARD_TRANSFORM_BLOCK_SIZE) : 1u;
  var transformed = hadamard_butterflies(value, lanes);
  if (lanes < HADAMARD_TRANSFORM_BLOCK_SIZE) {
    typealias S = WorkgroupScratch<float, THREADS>;
    let invocation = subgroup_invocation();
    S.data[invocation] = transformed;
    GroupMemoryBarrierWithGroupSync();
    // Element j of `column` is the value at the block's lane position of this invocation in its j-th subgroup.
    let count = HADAMARD_TRANSFORM_BLOCK_SIZE / lanes;
    let first = invocation & ~(HADAMARD_TRANSFORM_BLOCK_SIZE - lanes);
    float column[HADAMARD_TRANSFORM_BLOCK_SIZE];
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
    transformed = column[(invocation % HADAMARD_TRANSFORM_BLOCK_SIZE) / lanes];
    GroupMemoryBarrierWithGroupSync();
  }
  return transformed / sqrt(float(HADAMARD_TRANSFORM_BLOCK_SIZE));
}
