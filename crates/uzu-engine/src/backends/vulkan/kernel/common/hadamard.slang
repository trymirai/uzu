import common.activation;
import generated.hadamard_order;

// Ping-pong Hadamard butterfly stages, one element per invocation of workgroups of up to 256 invocations.
groupshared float hadamard_stages[2][256];

// A random Hadamard factor applied as the CPU's FP32 product: canonical signs flip the sign bit, keeping subnormals.
public float hadamard_factor(float value, int factor) {
  if (factor == 1 || factor == -1) {
    return asfloat(asuint(value) ^ (factor < 0 ? 0x80000000u : 0u));
  }
  return exact_product(value, float(factor));
}

// a + b or a - b rounded to nearest even, subnormals kept. When both magnitudes are below 2^-101 both are scaled up by
// 2^24 exactly, so the native operation sees normals: a sum below the smallest normal is exact on the 2^-149 grid and a
// larger one rounds as the scaled one, so scaling back down is exact. Otherwise any subnormal operand is below half the
// other's ULP (at least 2^-125 below a power of two), leaving the rounded sum unchanged, and a nonzero cancellation of
// operands from 2^-101 on is at least the smallest normal.
float hadamard_sum(float a, float b, bool subtract) {
  // A branch, not flattened: the scaled path is rare and would otherwise run for every sum.
  [branch]
  if (max(asuint(a) & 0x7fffffffu, asuint(b) & 0x7fffffffu) < 0x0d000000u) {
    let scaled_a = exact_scale_up_24(a);
    let scaled_b = exact_scale_up_24(b);
    return exact_scale_down(subtract ? scaled_a - scaled_b : scaled_a + scaled_b, 24);
  }
  return subtract ? a - b : a + b;
}

// Hadamard transform of logical blocks of HADAMARD_TRANSFORM_BLOCK_SIZE consecutive invocations of a workgroup of
// THREADS invocations, as the CPU hadamard_transform: the FP32 butterfly, then the product with the FP32 reciprocal of
// sqrt(block size), 0x3e3504f3 for 32. Blocks live in workgroup shared memory, so they do not depend on the device
// subgroup width. Every invocation must call it.
public float hadamard_transform<let THREADS : uint>(float value, uint tid) {
  static_assert(HADAMARD_TRANSFORM_BLOCK_SIZE == 32, "the scale is the FP32 reciprocal of sqrt(32)");
  static_assert(THREADS % HADAMARD_TRANSFORM_BLOCK_SIZE == 0, "Hadamard transform blocks must tile the workgroup");
  static_assert(THREADS <= 256, "the workgroup must fit the Hadamard scratch");
  hadamard_stages[0][tid] = value;
  GroupMemoryBarrierWithGroupSync();
  uint stage = 0;
  for (uint stride = 1; stride < HADAMARD_TRANSFORM_BLOCK_SIZE; stride <<= 1) {
    let own = hadamard_stages[stage][tid];
    let partner = hadamard_stages[stage][tid ^ stride];
    stage ^= 1;
    hadamard_stages[stage][tid] = hadamard_sum(partner, own, (tid & stride) != 0);
    GroupMemoryBarrierWithGroupSync();
  }
  let transformed = hadamard_stages[stage][tid];
  // An even stage count ends in stage 0, which the next call overwrites: every invocation reads first.
  GroupMemoryBarrierWithGroupSync();
  return exact_product(transformed, asfloat(0x3e3504f3u));
}
