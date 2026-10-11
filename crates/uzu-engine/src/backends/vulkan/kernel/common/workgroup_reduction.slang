import definitions;
import common.activation;

// One entry per subgroup; 256 covers the smallest possible subgroup width of the largest workgroup.
groupshared float workgroup_sum_scratch[256];
groupshared uint workgroup_magnitude_scratch[256];

// The native sum over the workgroup as Softmax combines its pairs: subgroup sums, one shared entry per subgroup, then
// subgroup 0 sums the entries; with it the largest magnitude bits of any value, NaN above infinity above finite values.
// It assumes nothing about the subgroup width or which invocations form a subgroup. Every invocation must call it, and
// all read the same results.
float workgroup_native_sum(float value, out uint largest) {
  let subgroup_total = WaveActiveSum(value);
  let subgroup_largest = WaveActiveMax(asuint(value) & 0x7fffffffu);
  if (WavePrefixSum(1u) == 0) {
    workgroup_sum_scratch[subgroup_id()] = subgroup_total;
    workgroup_magnitude_scratch[subgroup_id()] = subgroup_largest;
  }
  GroupMemoryBarrierWithGroupSync();
  if (subgroup_id() == 0) {
    // Active lanes stride over every subgroup entry, even when there are more subgroups than lanes.
    let lane = WavePrefixSum(1u);
    let lanes = WaveActiveSum(1u);
    float total = 0.0;
    uint magnitude = 0;
    for (uint subgroup = lane; subgroup < subgroup_count(); subgroup += lanes) {
      total += workgroup_sum_scratch[subgroup];
      magnitude = max(magnitude, workgroup_magnitude_scratch[subgroup]);
    }
    total = WaveActiveSum(total);
    magnitude = WaveActiveMax(magnitude);
    if (WavePrefixSum(1u) == 0) {
      workgroup_sum_scratch[0] = total;
      workgroup_magnitude_scratch[0] = magnitude;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  let total = workgroup_sum_scratch[0];
  largest = workgroup_magnitude_scratch[0];
  // Every invocation has read the results before another sum reuses the scratch.
  GroupMemoryBarrierWithGroupSync();
  return total;
}

// Sums over a workgroup of THREADS invocations. Natively unless every value is finite and below 2^-101 in magnitude,
// where the device would flush subnormal operands and sums: then, a uniform decision on the shared largest magnitude,
// every value is scaled up by 2^24 exactly, so all are multiples of 2^-125 below 2^-77 whose sums, cancellations
// included, are normal or zero and round as unscaled IEEE sums do, and the scaled sum comes back down exactly or with
// one rounding onto the subnormal grid. Every invocation must call it.
public float workgroup_sum<let THREADS : uint>(float value) {
  static_assert(THREADS <= 256, "the workgroup must fit the subgroup scratch");
  uint largest;
  let total = workgroup_native_sum(value, largest);
  [branch]
  if (largest >= 0x0d000000u) {
    return total;
  }
  return exact_scale_down(workgroup_native_sum(exact_scale_up_24(value), largest), 24);
}

// partial + x x for a non-negative partial of earlier squares, as the CPU's separately rounded square and sum with
// subnormals kept. Natively, both operations NoContraction, where the partial is at least 2^-101 or |x| at least 2^-50
// (infinities and NaN included): a square or partial the device may flush is then below 2^-126, under half the other
// addend's upward FP32 spacing of at least 2^-124, so it cannot change the rounded sum, and the other is normal;
// otherwise through exact_product and exact_sum.
public float accumulate_squares(float partial, float x) {
  [branch]
  if (asuint(partial) >= 0x0d000000u || (asuint(x) & 0x7fffffffu) >= 0x26800000u) {
    return spirv_asm {
      %square:$$float = OpFMul $x $x;
      OpDecorate %square NoContraction;
      %sum:$$float = OpFAdd $partial %square;
      OpDecorate %sum NoContraction;
      result:$$float = OpCopyObject %sum;
    };
  }
  return exact_sum(partial, exact_product(x, x), false);
}

// sum / count by the FP32 count as the CPU divides: natively where the quotient is normal; where it may not be
// (exponent fields at least 126 apart, so |sum| < 2^-93 for any count and scaling it is exact), the numerator scaled
// up by 2^24 and the quotient back down, one rounding onto the subnormal grid. Zeros return as they are, so their sign
// never depends on the division; NaN and infinities divide natively. The division is within Vulkan's 2.5 ULPs, of the
// scaled quotient below the smallest normal.
public float staged_mean(float sum, uint count) {
  let divisor = float(count);
  if ((asuint(sum) & 0x7fffffffu) == 0) {
    return sum;
  }
  [branch]
  if (((asuint(sum) >> 23) & 0xffu) + 126 <= ((asuint(divisor) >> 23) & 0xffu)) {
    return exact_scale_down(exact_scale_up_24(sum) / divisor, 24);
  }
  return sum / divisor;
}

// The CPU's reciprocal RMS 1 / sqrt(sum / count + epsilon) over a workgroup of THREADS invocations from each one's
// partial sum of squares: workgroup_sum's native reduction, in its own order, whose possibly flushed subnormal partials
// are negligible beside a partial from 2^-101, or its exact scaled sum of tiny partials; the epsilon sum exact with
// subnormals kept; staged_mean's division within Vulkan's 2.5 ULPs; the builtin reciprocal square root of positive
// normal values within Vulkan's 2 ULPs. The other classes are decided on the bits as the CPU's square root and
// reciprocal give them: zeros give infinities of their sign, +inf gives +0, negative values and NaN give NaN, and a
// positive subnormal is scaled up by 2^24 with the result scaled by 2^12 exactly. Every invocation must call it.
public float rms_inverse<let THREADS : uint>(float partial, uint count, float epsilon) {
  let shifted = exact_sum(staged_mean(workgroup_sum<THREADS>(partial), count), epsilon, false);
  let bits = asuint(shifted);
  if (bits >= 0x00800000u && bits < 0x7f800000u) {
    return rsqrt(shifted);
  }
  if ((bits & 0x7fffffffu) == 0) {
    return asfloat(bits | 0x7f800000u);
  }
  if (bits == 0x7f800000u) {
    return 0.0;
  }
  if (bits < 0x00800000u) {
    return rsqrt(exact_scale_up_24(shifted)) * 4096.0;
  }
  return asfloat(0x7fc00000u);
}
