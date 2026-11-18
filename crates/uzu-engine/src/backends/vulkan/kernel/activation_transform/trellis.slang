import definitions;
import common.activation_quantization;
import common.hadamard;
import common.workgroup_reduction;
import generated.trellis;

static const uint TRELLIS_THREADS = 128;
static const uint TRELLIS_CAPACITY = TRELLIS_THREADS * 16;

groupshared float trellis_hadamard[TRELLIS_CAPACITY];

void trellis_hadamard_transform<let H : uint, let VALUES : uint>(inout float values[VALUES], uint tid) {
  static_assert(H == 512 || H == 1024 || H == 2048 || H == 4096, "a normalized block");
  static_assert(VALUES == (H == 4096 ? 32u : 16u), "16 values per invocation, 32 for H 4096");
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
      GroupMemoryBarrierWithGroupSync();
    }
  }
  let normalization = 1.0 / sqrt(float(H));
  [ForceUnroll]
  for (uint k = 0; k < VALUES; k++) {
    values[k] *= normalization;
  }
}

void trellis_row<let D : uint>(
  Ptr<bf16, Access.Read> input,
  Ptr<int, Access.Read> rht_factors,
  Ptr<float, Access.Read> mixing,
  Ptr<int8_t, Access.ReadWrite> activations,
  Ptr<float, Access.ReadWrite> column_group_sums,
  Ptr<float, Access.ReadWrite> scales,
  uint token,
) {
  // H is the power-of-two factor of D: one H-point transform for each of the M odd columns.
  static const uint H = D & (0u - D);
  static const uint M = D / H;
  static const uint VALUES = H == 4096 ? 32u : 16u;
  static const uint COLUMNS = TRELLIS_THREADS * VALUES / H;
  static const uint CALLS = (M + COLUMNS - 1) / COLUMNS;
  let row = token * D;
  let tid = subgroup_invocation();
  let invocations = H / VALUES;
  let u = tid % invocations;
  uint retained[CALLS * VALUES / 2] = {};
  uint largest = 0;
  for (uint call = 0; call < CALLS; call++) {
    let o = call * COLUMNS + tid / invocations;
    float values[VALUES];
    for (uint k = 0; k < VALUES; k++) {
      let h = u * VALUES + k;
      values[k] = 0.0;
      if (o < M) {
        for (uint i = 0; i < M; i++) {
          values[k] = fma(input[row + h * M + i].toFloat() * float(rht_factors[h * M + i]), mixing[o * M + i], values[k]);
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
  let row_scale = activation_quantization_scale(workgroup_reduce<Max<uint>, TRELLIS_THREADS>(largest << 16));
  int partial[COLUMN_GROUP_COUNT] = {};
  for (uint call = 0; call < CALLS; call++) {
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
[[Precondition("new", "dimension == 2048 || dimension == 2560 || dimension == 3072 || dimension == 3584 || dimension == 4096 || dimension == 5120 || dimension == 6144 || dimension == 6656 || dimension == 7168 || dimension == 9216 || dimension == 10240 || dimension == 12288 || dimension == 14336 || dimension == 17408 || dimension == 18944 || dimension == 19968")]]
func TrellisTransform(
  Ptr<bf16, Access.Read> input,
  Ptr<int, Access.Read> rht_factors,
  Ptr<float, Access.Read> mixing,
  Ptr<int8_t, Access.ReadWrite> activations,
  Ptr<float, Access.ReadWrite> column_group_sums,
  Ptr<float, Access.ReadWrite> scales,
  uint batch,
  [[Groups("batch")]] uint token_group,
  [[Threads("128")]] uint tid,
  [[Specialize]] uint dimension,
) {
  static_assert(TRELLIS_THREADS == 128, "Threads(128)");
  switch (dimension) {
  case 2048:
    return trellis_row<2048>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 2560:
    return trellis_row<2560>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 3072:
    return trellis_row<3072>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 3584:
    return trellis_row<3584>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 4096:
    return trellis_row<4096>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 5120:
    return trellis_row<5120>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 6144:
    return trellis_row<6144>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 6656:
    return trellis_row<6656>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 7168:
    return trellis_row<7168>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 9216:
    return trellis_row<9216>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 10240:
    return trellis_row<10240>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 12288:
    return trellis_row<12288>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 14336:
    return trellis_row<14336>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 17408:
    return trellis_row<17408>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 18944:
    return trellis_row<18944>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  case 19968:
    return trellis_row<19968>(input, rht_factors, mixing, activations, column_group_sums, scales, token_group);
  }
}
