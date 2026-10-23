import definitions;
import common.activation_quantization;
import common.workgroup_reduction;
import generated.trellis;

// TrellisTransform of token rows of D = H M columns, H the power-of-two factor of D (the Hadamard block) and M odd.
// Output column o < M is the fma chain over i ascending from +0 of input[h M + i] factor[h M + i] mixing[o M + i], then
// the FP32 block transform, rounded to BF16 and quantized; columns past M hold +0 and store nothing.

static const uint TRELLIS_THREADS = 128;
static const uint TRELLIS_CAPACITY = 2048;

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
  for (uint stride = 1; stride < lanes; stride <<= 1) {
    let high = (WaveGetLaneIndex() & stride) != 0;
    for (uint k = 0; k < VALUES; k++) {
      let partner = shuffle_xor(values[k], stride);
      values[k] = high ? partner - values[k] : values[k] + partner;
    }
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

// One row of D = H M columns: VALUES elements per invocation and CALLS = 2 WORDS / VALUES transforms, retaining
// WORDS words of BF16 pairs.
void trellis_row<let D : uint, let H : uint, let M : uint, let VALUES : uint, let WORDS : uint>(
  Ptr<bf16, Access.Read> input,
  Ptr<float, Access.Read> rht_factors,
  Ptr<float, Access.Read> mixing,
  Ptr<int8_t, Access.ReadWrite> activations,
  Ptr<float, Access.ReadWrite> column_group_sums,
  Ptr<float, Access.ReadWrite> scales,
  uint token,
  uint tid,
) {
  static_assert(H * M == D && H == (D & (0u - D)) && M % 2 == 1, "H is the power-of-two factor of D, M odd");
  static_assert(VALUES % 2 == 0 && TRELLIS_THREADS * VALUES % H == 0, "whole BF16 pairs and whole blocks");
  static_assert(2 * WORDS == (M + TRELLIS_THREADS * VALUES / H - 1) / (TRELLIS_THREADS * VALUES / H) * VALUES,
                "WORDS retain every call's values");
  let row = token * D;
  let invocations = H / VALUES;
  let columns = TRELLIS_THREADS / invocations;
  let u = tid % invocations;
  let calls = 2 * WORDS / VALUES;
  uint retained[WORDS] = {};
  uint largest = 0;
  for (uint call = 0; call < calls; call++) {
    let o = call * columns + tid / invocations;
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
      retained[(call * VALUES + k) / 2] |= bits << (k % 2 * 16);
      let magnitude = bits & 0x7fffu;
      largest = max(largest, magnitude > 0x7f80u ? 0u : magnitude);
    }
  }
  // BF16 magnitude bits widen exactly.
  let row_scale = activation_quantization_scale(workgroup_reduce<Max<uint>, TRELLIS_THREADS>(largest << 16));
  int partial[COLUMN_GROUP_COUNT] = {};
  for (uint call = 0; call < calls; call++) {
    let o = call * columns + tid / invocations;
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
[[Precondition("new", "dimension == 2048 || dimension == 2560 || dimension == 3072 || dimension == 3584 || dimension == 4096 || dimension == 5120 || dimension == 6144 || dimension == 6656 || dimension == 7168 || dimension == 9216 || dimension == 10240 || dimension == 12288 || dimension == 14336 || dimension == 17408 || dimension == 18944 || dimension == 19968")]]
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
  switch (dimension) {
  case 2048:
    trellis_row<2048, 2048, 1, 16, 8>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                      token_group, subgroup_invocation());
    break;
  case 2560:
    trellis_row<2560, 512, 5, 16, 16>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                      token_group, subgroup_invocation());
    break;
  case 3072:
    trellis_row<3072, 1024, 3, 16, 16>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                       token_group, subgroup_invocation());
    break;
  case 3584:
    trellis_row<3584, 512, 7, 16, 16>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                      token_group, subgroup_invocation());
    break;
  case 4096:
    trellis_row<4096, 4096, 1, 32, 16>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                       token_group, subgroup_invocation());
    break;
  case 5120:
    trellis_row<5120, 1024, 5, 16, 24>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                       token_group, subgroup_invocation());
    break;
  case 6144:
    trellis_row<6144, 2048, 3, 16, 24>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                       token_group, subgroup_invocation());
    break;
  case 6656:
    trellis_row<6656, 512, 13, 16, 32>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                       token_group, subgroup_invocation());
    break;
  case 7168:
    trellis_row<7168, 1024, 7, 16, 32>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                       token_group, subgroup_invocation());
    break;
  case 9216:
    trellis_row<9216, 1024, 9, 16, 40>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                       token_group, subgroup_invocation());
    break;
  case 10240:
    trellis_row<10240, 2048, 5, 16, 40>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                        token_group, subgroup_invocation());
    break;
  case 12288:
    trellis_row<12288, 4096, 3, 32, 48>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                        token_group, subgroup_invocation());
    break;
  case 14336:
    trellis_row<14336, 2048, 7, 16, 56>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                        token_group, subgroup_invocation());
    break;
  case 17408:
    trellis_row<17408, 1024, 17, 16, 72>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                         token_group, subgroup_invocation());
    break;
  case 18944:
    trellis_row<18944, 512, 37, 16, 80>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                        token_group, subgroup_invocation());
    break;
  case 19968:
    trellis_row<19968, 512, 39, 16, 80>(input, rht_factors, mixing, activations, column_group_sums, scales,
                                        token_group, subgroup_invocation());
    break;
  }
}
