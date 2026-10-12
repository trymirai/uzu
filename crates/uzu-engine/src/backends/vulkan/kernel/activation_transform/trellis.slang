import definitions;
import common.activation;
import common.activation_quantization;
import common.storage_address;
import generated.trellis;

// The CPU TrellisTransform of each token row of DIMENSION = H M columns, H = DIMENSION & -DIMENSION the Hadamard block
// and M = DIMENSION / H the odd mixing dimension. Output column o < M mixes, for each h < H, the products of the widened
// BF16 input and the factor at h M + i with mixing[o M + i], i ascending from +0, each product an exact_product and each
// step one exact_fma rounding as the CPU's mul_add; then the FP32 transform of block H and the BF16 rounding of every
// result, retained two per word, give the scale, the codes and their sums by column mod COLUMN_GROUP_COUNT through the
// canonical activation quantizer. A workgroup of 128 invocations transforms 128 / P columns per call, P = H / VALUES
// invocations each; columns past M hold +0 and store nothing. Workgroups stride over tokens by the actual NumWorkgroups,
// which the host clamps to the device limit; rows are addressed in 64-bit words, so no index wraps.

static const uint TRELLIS_THREADS = 128;
static const uint TRELLIS_CAPACITY = 2048;

groupshared float trellis_hadamard[TRELLIS_CAPACITY];

// The CPU hadamard_transform of block H: on entry and exit values[k] is element u VALUES + k of block tid / P,
// u = tid % P. Element bits below log2(VALUES) pair in registers; each of VALUES / 16 passes stores values
// pass 16 + k at slots tid 16 + k, so the remaining bits pair in trellis_hadamard, eight disjoint pairs per invocation.
// Every invocation must call it.
void trellis_hadamard_transform<let H : uint, let VALUES : uint>(inout float values[VALUES], uint tid) {
  static_assert(H == 512 || H == 1024 || H == 2048 || H == 4096, "a normalized block");
  static_assert(VALUES == (H == 4096 ? 32u : 16u), "16 values per invocation, 32 for H 4096");
  static_assert(TRELLIS_THREADS * 16 == TRELLIS_CAPACITY, "one pass fills the scratch");
  for (uint stride = 1; stride < VALUES; stride <<= 1) {
    for (uint pair = 0; pair < VALUES / 2; pair++) {
      let lo = ((pair & ~(stride - 1)) << 1) | (pair & (stride - 1));
      let a = values[lo];
      let b = values[lo | stride];
      values[lo] = exact_sum(a, b, false);
      values[lo | stride] = exact_sum(a, b, true);
    }
  }
  [ForceUnroll]
  for (uint pass = 0; pass < VALUES / 16; pass++) {
    [ForceUnroll]
    for (uint k = 0; k < 16; k++) {
      trellis_hadamard[tid * 16 + k] = values[pass * 16 + k];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 16; stride < 16 * (H / VALUES); stride <<= 1) {
      [ForceUnroll]
      for (uint ordinal = 0; ordinal < 8; ordinal++) {
        let item = tid + ordinal * TRELLIS_THREADS;
        let lo = ((item & ~(stride - 1)) << 1) | (item & (stride - 1));
        let a = trellis_hadamard[lo];
        let b = trellis_hadamard[lo | stride];
        trellis_hadamard[lo] = exact_sum(a, b, false);
        trellis_hadamard[lo | stride] = exact_sum(a, b, true);
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
  let normalization = H == 512 ? 0x3d3504f3u : H == 1024 ? 0x3d000000u : H == 2048 ? 0x3cb504f3u : 0x3c800000u;
  [ForceUnroll]
  for (uint k = 0; k < VALUES; k++) {
    values[k] = exact_product(values[k], asfloat(normalization));
  }
}

// One row of D = H M columns: VALUES elements per invocation and CALLS = 2 WORDS / VALUES transforms, retaining
// WORDS words of BF16 pairs.
void trellis_row<let D : uint, let H : uint, let M : uint, let VALUES : uint, let WORDS : uint>(
  Ptr<bf16, Access.Read> input,
  Ptr<float, Access.Read> rht_factors,
  Ptr<float, Access.Read> mixing,
  Ptr<int8_t, Access.ReadWrite> codes,
  Ptr<float, Access.ReadWrite> sums,
  Ptr<float, Access.ReadWrite> scale,
  uint tid,
) {
  static_assert(H * M == D && H == (D & (0u - D)) && M % 2 == 1, "H is the power-of-two factor of D, M odd");
  static_assert(VALUES % 2 == 0 && TRELLIS_THREADS * VALUES % H == 0, "whole BF16 pairs and whole blocks");
  static_assert(2 * WORDS == (M + TRELLIS_THREADS * VALUES / H - 1) / (TRELLIS_THREADS * VALUES / H) * VALUES,
                "WORDS retain every call's values");
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
          let product = exact_product(input[h * M + i].toFloat(), rht_factors[h * M + i]);
          values[k] = exact_fma(product, mixing[o * M + i], values[k]);
        }
      }
    }
    trellis_hadamard_transform<H, VALUES>(values, tid);
    for (uint k = 0; k < VALUES; k++) {
      let bits = uint(bf16.fromFloat(values[k]).bits);
      retained[(call * VALUES + k) / 2] |= bits << (k % 2 * 16);
      let magnitude = bits & 0x7fffu;
      largest = max(largest, magnitude > 0x7f80u ? 0u : magnitude);
    }
  }
  // BF16 magnitude bits widen exactly; the reduction maps 0 and infinity to 1, otherwise the maximum over 127.
  let row_scale = asfloat(activation_quantization_reduce(largest << 16, tid, TRELLIS_THREADS, false));
  int partial[COLUMN_GROUP_COUNT] = {};
  for (uint call = 0; call < calls; call++) {
    let o = call * columns + tid / invocations;
    for (uint k = 0; o < M && k < VALUES; k++) {
      let column = (u * VALUES + k) * M + o;
      let bits = retained[(call * VALUES + k) / 2] >> (k % 2 * 16) << 16;
      let code = activation_quantization_code(asfloat(bits), row_scale);
      codes[column] = int8_t(code);
      partial[column % COLUMN_GROUP_COUNT] += code;
    }
  }
  for (uint column_group = 0; column_group < COLUMN_GROUP_COUNT; column_group++) {
    let sum = int(activation_quantization_reduce(uint(partial[column_group]), tid, TRELLIS_THREADS, true));
    if (tid == 0) {
      sums[column_group] = float(sum);
    }
  }
  if (tid == 0) {
    scale[0] = row_scale;
  }
}

[[Kernel]] [[Public]]
[[Precondition("new", "DIMENSION == 4096 || DIMENSION == 5120 || DIMENSION == 6144 || DIMENSION == 6656 || DIMENSION == 17408 || DIMENSION == 19968")]]
func TrellisTransform(
  Ptr<bf16, Access.Read> input,
  Ptr<float, Access.Read> rht_factors,
  Ptr<float, Access.Read> mixing,
  Ptr<int8_t, Access.ReadWrite> activations,
  Ptr<float, Access.ReadWrite> column_group_sums,
  Ptr<float, Access.ReadWrite> scales,
  uint batch,
  [[Groups("batch.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0])")]] uint token_group,
  [[Threads("128")]] uint tid,
  [[Specialize]] uint DIMENSION,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  var token = token_group;
  while (token < batch) {
    let row = element_index(token, DIMENSION, 0);
    let source = element_pointer(input, row);
    let codes = element_pointer(activations, row);
    let sums = element_pointer(column_group_sums, element_index(token, COLUMN_GROUP_COUNT, 0));
    let scale = element_pointer(scales, uint2(token, 0));
    switch (DIMENSION) {
    case 4096:
      trellis_row<4096, 4096, 1, 32, 16>(source, rht_factors, mixing, codes, sums, scale, tid);
      break;
    case 5120:
      trellis_row<5120, 1024, 5, 16, 24>(source, rht_factors, mixing, codes, sums, scale, tid);
      break;
    case 6144:
      trellis_row<6144, 2048, 3, 16, 24>(source, rht_factors, mixing, codes, sums, scale, tid);
      break;
    case 6656:
      trellis_row<6656, 512, 13, 16, 32>(source, rht_factors, mixing, codes, sums, scale, tid);
      break;
    case 17408:
      trellis_row<17408, 1024, 17, 16, 72>(source, rht_factors, mixing, codes, sums, scale, tid);
      break;
    case 19968:
      trellis_row<19968, 512, 39, 16, 80>(source, rht_factors, mixing, codes, sums, scale, tid);
      break;
    }
    if (batch - token <= groups.x) {
      break;
    }
    token += groups.x;
  }
}
