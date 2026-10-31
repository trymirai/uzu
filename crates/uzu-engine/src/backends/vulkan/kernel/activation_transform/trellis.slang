import definitions;
import common.activation_quantization;
import common.hadamard;
import common.workgroup_reduction;
import generated.trellis;

// TrellisTransform of token rows of D = H M columns, H the power-of-two factor of D (the Hadamard block) and M odd.
// Output column o < M is the fma chain over i ascending from +0 of input[h M + i] factor[h M + i] mixing[o M + i], then
// the FP32 block transform, rounded to BF16 and quantized; columns past M hold +0 and store nothing.

static const uint TRELLIS_THREADS = 128;
static const uint TRELLIS_CAPACITY = 2048;
// Metal's MAX_DIMENSION, repeated by the TrellisTransform precondition; the retained words are sized for it.
static const uint TRELLIS_MAX_DIMENSION = 20480;
static const uint TRELLIS_MAX_WORDS = TRELLIS_MAX_DIMENSION / (2 * TRELLIS_THREADS);

groupshared float trellis_hadamard[TRELLIS_CAPACITY];

// The normalized transform of block H: values[k] is element u VALUES + k of block tid / P, u = tid % P and P = H /
// VALUES, on entry and exit. Strides below VALUES pair in registers, below the subgroup width by shuffle_xor, and the
// rest in trellis_hadamard. Every invocation must call it.
void trellis_hadamard_transform<let H : uint, let VALUES : uint>(inout float values[VALUES], uint tid) {
  static_assert(H == 512 || H == 1024 || H == 2048 || H == 4096, "a normalized block");
  static_assert(VALUES == (H == 4096 ? 32u : 16u), "16 values per invocation, 32 for H 4096");
  static_assert(TRELLIS_THREADS * 16 == TRELLIS_CAPACITY, "one pass fills the scratch");
  for (uint stride = 1; stride < VALUES; stride <<= 1) {
    for (uint pair = 0; pair < VALUES / 2; pair++) {
      let lo = ((pair & ~(stride - 1)) << 1) | (pair & (stride - 1));
      let a = values[lo];
      let b = values[lo | stride];
      values[lo] = a + b;
      values[lo | stride] = a - b;
    }
  }
  let lanes = min(WaveGetLaneCount(), H / VALUES);
  for (uint k = 0; k < VALUES; k++) {
    values[k] = hadamard_butterflies(values[k], lanes);
  }
  if (lanes < H / VALUES) {
    [ForceUnroll]
    for (uint pass = 0; pass < VALUES / 16; pass++) {
      [ForceUnroll]
      for (uint k = 0; k < 16; k++) {
        trellis_hadamard[tid * 16 + k] = values[pass * 16 + k];
      }
      GroupMemoryBarrierWithGroupSync();
      for (uint stride = 16 * lanes; stride < 16 * (H / VALUES); stride <<= 1) {
        [ForceUnroll]
        for (uint ordinal = 0; ordinal < 8; ordinal++) {
          let item = tid + ordinal * TRELLIS_THREADS;
          let lo = ((item & ~(stride - 1)) << 1) | (item & (stride - 1));
          let a = trellis_hadamard[lo];
          let b = trellis_hadamard[lo | stride];
          trellis_hadamard[lo] = a + b;
          trellis_hadamard[lo | stride] = a - b;
        }
        GroupMemoryBarrierWithGroupSync();
      }
      [ForceUnroll]
      for (uint k = 0; k < 16; k++) {
        values[pass * 16 + k] = trellis_hadamard[tid * 16 + k];
      }
      // Every invocation reads its slots before the next pass or call stores again.
      GroupMemoryBarrierWithGroupSync();
    }
  }
  let normalization = 1.0 / sqrt(float(H));
  [ForceUnroll]
  for (uint k = 0; k < VALUES; k++) {
    values[k] *= normalization;
  }
}

// One row of D = H M columns: VALUES elements per invocation and ceil(M / COLUMNS) transforms, retaining their values
// as BF16 pairs.
void trellis_row<let H : uint, let VALUES : uint>(
  Ptr<bf16, Access.Read> input,
  Ptr<float, Access.Read> rht_factors,
  Ptr<float, Access.Read> mixing,
  Ptr<int8_t, Access.ReadWrite> activations,
  Ptr<float, Access.ReadWrite> column_group_sums,
  Ptr<float, Access.ReadWrite> scales,
  uint dimension,
  uint token,
  uint tid,
) {
  static const uint COLUMNS = TRELLIS_THREADS * VALUES / H;
  static_assert(VALUES % 2 == 0 && TRELLIS_THREADS * VALUES % H == 0, "whole BF16 pairs and whole blocks");
  static_assert((TRELLIS_MAX_DIMENSION / H + COLUMNS - 1) / COLUMNS * VALUES <= 2 * TRELLIS_MAX_WORDS,
                "the retained words hold every call's values");
  let M = dimension / H;
  let row = token * dimension;
  let invocations = H / VALUES;
  let u = tid % invocations;
  let calls = (M + COLUMNS - 1) / COLUMNS;
  uint retained[TRELLIS_MAX_WORDS];
  uint largest = 0;
  for (uint call = 0; call < calls; call++) {
    let o = call * COLUMNS + tid / invocations;
    float values[VALUES];
    for (uint k = 0; k < VALUES; k++) {
      let h = u * VALUES + k;
      values[k] = 0.0;
      if (o < M) {
        for (uint i = 0; i < M; i++) {
          values[k] = fma(input[row + h * M + i].toFloat() * rht_factors[h * M + i], mixing[o * M + i], values[k]);
        }
      }
    }
    trellis_hadamard_transform<H, VALUES>(values, tid);
    for (uint k = 0; k < VALUES; k++) {
      let bits = bf16.fromFloat(values[k]).storageBits();
      let word = (call * VALUES + k) / 2;
      if (k % 2 == 0) {
        retained[word] = bits;
      } else {
        retained[word] |= bits << 16;
      }
      let magnitude = bits & 0x7fffu;
      largest = max(largest, magnitude > 0x7f80u ? 0u : magnitude);
    }
  }
  // BF16 magnitude bits widen exactly.
  let row_scale = activation_quantization_scale(workgroup_reduce<Max<uint>, TRELLIS_THREADS>(largest << 16));
  int partial[COLUMN_GROUP_COUNT] = {};
  for (uint call = 0; call < calls; call++) {
    let o = call * COLUMNS + tid / invocations;
    for (uint k = 0; o < M && k < VALUES; k++) {
      let column = (u * VALUES + k) * M + o;
      let bits = retained[(call * VALUES + k) / 2] >> (k % 2 * 16) << 16;
      let code = activation_quantization_code(asfloat(bits), row_scale);
      activations[row + column] = int8_t(code);
      partial[column % COLUMN_GROUP_COUNT] += code;
    }
  }
  for (uint column_group = 0; column_group < COLUMN_GROUP_COUNT; column_group++) {
    let sum = workgroup_reduce<Sum<int>, TRELLIS_THREADS>(partial[column_group]);
    if (tid == 0) {
      column_group_sums[token * COLUMN_GROUP_COUNT + column_group] = float(sum);
    }
  }
  if (tid == 0) {
    scales[token] = row_scale;
  }
}

[[Kernel]] [[Public]]
[[Precondition("new", "dimension <= 20480 && (9..=12).contains(&dimension.trailing_zeros())")]]
func TrellisTransform(
  Ptr<bf16, Access.Read> input,
  Ptr<float, Access.Read> rht_factors,
  Ptr<float, Access.Read> mixing,
  Ptr<int8_t, Access.ReadWrite> activations,
  Ptr<float, Access.ReadWrite> column_group_sums,
  Ptr<float, Access.ReadWrite> scales,
  uint batch,
  [[Groups("batch")]] uint token_group,
  [[Threads("128")]] uint tid,
  [[Specialize]] uint dimension,
) {
  switch (dimension & (0u - dimension)) {
  case 512:
    trellis_row<512, 16>(input, rht_factors, mixing, activations, column_group_sums, scales, dimension, token_group,
                         subgroup_invocation());
    break;
  case 1024:
    trellis_row<1024, 16>(input, rht_factors, mixing, activations, column_group_sums, scales, dimension, token_group,
                          subgroup_invocation());
    break;
  case 2048:
    trellis_row<2048, 16>(input, rht_factors, mixing, activations, column_group_sums, scales, dimension, token_group,
                          subgroup_invocation());
    break;
  case 4096:
    trellis_row<4096, 32>(input, rht_factors, mixing, activations, column_group_sums, scales, dimension, token_group,
                          subgroup_invocation());
    break;
  }
}
