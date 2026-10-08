import common.activation;

// Symmetric INT8 activation quantization of a tile of ACTIVATION_QUANTIZATION_TILE consecutive row elements, one per
// invocation, as the CPU quantize_transformed_row. Scale and sum groups are powers of two up to the tile, so groups never
// straddle tiles; inactive invocations past the row hold neutral zeros. Every invocation must call it.

public static const uint ACTIVATION_QUANTIZATION_TILE = 128;

// Ping-pong reduction stages, one element per invocation.
groupshared uint activation_quantization_stages[2][ACTIVATION_QUANTIZATION_TILE];

// Over the aligned group of `group` consecutive invocations containing `tid`: the sum (wrapping) of `value`, or for
// magnitude bits the bits of the group's scale, which only the group's first invocation converts. Each invocation reads
// its own final slot before that one writes only its own slot of stage 1, so the final barrier, which every invocation
// passes before reading the scale, also publishes it; the next reduction writes stage 0 and passes a barrier before
// writing stage 1 again, after every scale read.
uint activation_quantization_reduce(uint value, uint tid, uint group, bool sum) {
  activation_quantization_stages[0][tid] = value;
  GroupMemoryBarrierWithGroupSync();
  uint stage = 0;
  for (uint stride = 1; stride < group; stride <<= 1) {
    let own = activation_quantization_stages[stage][tid];
    let partner = activation_quantization_stages[stage][tid ^ stride];
    stage ^= 1;
    activation_quantization_stages[stage][tid] = sum ? own + partner : max(own, partner);
    GroupMemoryBarrierWithGroupSync();
  }
  let result = activation_quantization_stages[stage][tid];
  let leader = tid & ~(group - 1);
  if (!sum && tid == leader) {
    let scale = result != 0u && result < 0x7f800000u ? activation_quantization_scale(result) : 1.0;
    activation_quantization_stages[1][tid] = asuint(scale);
  }
  // Every invocation reads before the next reduction overwrites the stages.
  GroupMemoryBarrierWithGroupSync();
  if (sum) {
    return result;
  }
  return activation_quantization_stages[1][leader];
}

// The bits of maximum / 127 rounded to nearest, from the bits of a positive finite maximum, exact where FP32 division is
// not: its significand times a power of two divided by 127 in 32-bit integers, the remainder rounding (127 is odd, so
// there are no ties). Results from the first normal binade down, and underflow to 0, are integers in units of 2^-149.
float activation_quantization_scale(uint maximum) {
  let exponent = maximum >> 23;
  let fraction = maximum & 0x007fffffu;
  var numerator = fraction;
  var bits = 0u;
  if (exponent > 7) {
    // A 24-bit quotient: shift 7 below 127 * 2^17, else 6; the numerator stays below 2^31.
    let significand = fraction | 0x00800000u;
    let shift = significand < (127u << 17) ? 7u : 6u;
    numerator = significand << shift;
    bits = (exponent - shift - 1) << 23;
  } else if (exponent > 0) {
    numerator = (fraction | 0x00800000u) << (exponent - 1);
  }
  let quotient = numerator / 127;
  // A carry rounds into the exponent.
  return asfloat(bits + quotient + (numerator - quotient * 127 > 63 ? 1u : 0u));
}

// The integer code of value / scale as the CPU computes it: the FP32 quotient rounded to nearest, then half away from
// zero, saturated to ±127, NaN to 0, classified on raw bits. Both operands are renormalized through their bits so the
// native division works on normal numbers with an unchanged quotient, after saturating ratios beyond the code range.
// Vulkan's division is within 2.5 ULPs, so it decides the code unless its quotient lies within 8 ordered FP32 steps of
// the nearest midpoint k + 1/2 (2.5 ULPs to the exact quotient plus half an ULP to its rounding, at most doubled across
// a binade). There an exact integer comparison decides whether the exact quotient reaches the midpoint's lower rounding
// boundary, halfway to its predecessor (an ULP below 0.5 is half the one above), the tie rounding to the even midpoint.
int activation_quantization_code(float value, float scale) {
  let value_bits = asuint(value) & 0x7fffffffu;
  let scale_bits = asuint(scale);
  var code = 0;
  if (value_bits > 0x7f800000u) {
    return 0;
  } else if (scale_bits == 0u || value_bits == 0x7f800000u) {
    // Nonzero over a scale that underflowed to 0, or infinite: infinite quotients; 0 / 0 is NaN.
    code = value_bits == 0u ? 0 : 127;
  } else if (value_bits != 0u) {
    int value_exponent;
    int scale_exponent;
    let value_significand = significand(value_bits, value_exponent);
    let scale_significand = significand(scale_bits, scale_exponent);
    // The quotient lies in (1/2, 2) 2^difference: from 8 on beyond 127.5, from -3 down below 1/4.
    let difference = value_exponent - scale_exponent;
    if (difference >= 8) {
      code = 127;
    } else if (difference >= -2) {
      let numerator = asfloat(uint(difference + 127) << 23 | (value_significand & 0x007fffffu));
      let denominator = asfloat(0x3f800000u | (scale_significand & 0x007fffffu));
      let quotient = numerator / denominator;
      let whole = floor(quotient);
      let midpoint = whole + 0.5;
      code = int(whole) + (quotient >= midpoint ? 1 : 0);
      // A branch, not flattened: the exact comparison is rare.
      [branch]
      if (abs(int(asuint(quotient)) - int(asuint(midpoint))) <= 8) {
        // The boundary is B 2^-shift with B = 2M - 1 for the midpoint's significand M, or 4M - 1 at the power of two
        // 1/2. The quotient reaches it when numerator >= boundary denominator, in integers V 2^(difference - 23) >= B 2^-shift
        // S 2^-23, so V 2^(difference + shift) >= B S, a 49-bit product.
        int midpoint_exponent;
        let midpoint_significand = significand(asuint(midpoint), midpoint_exponent);
        let power = midpoint_significand == 0x00800000u;
        let boundary = (midpoint_significand << (power ? 2 : 1)) - 1;
        let shift = uint(difference + (power ? 25 : 24) - midpoint_exponent);
        let product = multiply_extended(boundary, scale_significand);
        // The window bounds the shift to the significands' 23 to 27 bit gap.
        let high = value_significand >> (32 - shift);
        let low = value_significand << shift;
        let reaches = high > product.y || (high == product.y && low >= product.x);
        code = int(whole) + (reaches ? 1 : 0);
      }
      code = min(code, 127);
    }
  }
  return (asuint(value) & 0x80000000u) != 0 ? -code : code;
}

// Quantizes element `column` of row `row` of `count` columns, which tile `column / ACTIVATION_QUANTIZATION_TILE` covers:
// the scale is the group's largest magnitude, ignoring NaNs, over 127 when finite and positive, otherwise 1. Codes are
// stored at the Int8CodeLayout index: grouped by weight nibble, [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 4, 1, 5, 2, 6, 3, 7]
// within each 8 codes. Group sums add the codes of logical positions, before that permutation.
public void quantize_activation(
  float value,
  uint tid,
  uint row,
  uint column,
  uint count,
  Ptr<int8_t, Access.ReadWrite> codes,
  Ptr<float, Access.ReadWrite> scales,
  Ptr<int, Access.ReadWrite> group_sums,
  uint scale_group,
  uint sum_group,
  bool emit_group_sums,
  bool grouped_by_weight_nibble,
) {
  let active = column < count;
  let magnitude = asuint(value) & 0x7fffffffu;
  let scale = asfloat(activation_quantization_reduce(magnitude > 0x7f800000u ? 0u : magnitude, tid, scale_group, false));
  let code = activation_quantization_code(value, scale);
  if (active) {
    let offset = column % 8;
    codes[row * count + (grouped_by_weight_nibble ? column - offset + offset % 2 * 4 + offset / 2 : column)] =
      int8_t(code);
    if (column % scale_group == 0) {
      scales[(row * count + column) / scale_group] = scale;
    }
  }
  if (emit_group_sums) {
    let sum = int(activation_quantization_reduce(uint(code), tid, sum_group, true));
    if (active && column % sum_group == 0) {
      group_sums[(row * count + column) / sum_group] = sum;
    }
  }
}
