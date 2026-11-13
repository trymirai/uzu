import definitions;
import common.activation_quantization;
import common.hadamard;
import matmul.common.matmul;
import generated.hadamard_order;
import generated.quantization_method;
import matmul.common.int8_operands;
import matmul.common.loader;

using namespace linalg;

public interface ISharedWords {
  static uint4 read(uint word);
}

void add_k_slices<S : ISharedWords, let SLICES : int>(uint word, uint lane, out float values[8]) {
  [unroll]
  for (uint value = 0; value < 8; value++) {
    values[value] = 0.0;
  }
  [unroll]
  for (int slice = 0; slice < SLICES; slice++) {
    let low = S.read(word + uint(slice) * 64 + lane * 2);
    let high = S.read(word + uint(slice) * 64 + lane * 2 + 1);
    [unroll]
    for (uint value = 0; value < 4; value++) {
      values[value] = slice == 0 ? asfloat(low[value]) : values[value] + asfloat(low[value]);
      values[4 + value] = slice == 0 ? asfloat(high[value]) : values[4 + value] + asfloat(high[value]);
    }
  }
}

public interface ITensorShared : ISharedWords {
  static void write(uint word, uint4 value);
  static CoopMat<BFloat16, MemoryScope.Subgroup, 16, 16, USE> load<
    let USE : CoopMatMatrixUse,
    let LAYOUT : CoopMatMatrixLayout
  >(uint word, uint stride);
  static void store(
    CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value,
    uint word,
    uint stride
  );
}

public struct TensorCore<
  S : ITensorShared,
  let BM : int,
  let BN : int,
  let BK : int,
  let SG_M : int,
  let SG_N : int,
  let SG_K : int
> {
  static const int TM = BM / (16 * SG_M);
  static const int TN = BN / (16 * SG_N);
  static const uint STRIDE = uint(BK + 8) / 8;
  static const uint STAGE = uint(BM + BN) * STRIDE;

  CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> accumulators[TM][TN];
  uint row_word;
  uint col_word;
  uint slice;

  public __init() {
    let subgroup = subgroup_id();
    slice = subgroup % uint(SG_K);
    let region = subgroup / uint(SG_K);
    row_word = (region / uint(SG_N)) * uint(TM * 16) * STRIDE;
    col_word = uint(BM) * STRIDE + (region % uint(SG_N)) * uint(TN * 16) * STRIDE;
    [unroll]
    for (int i = 0; i < TM; i++) {
      [unroll]
      for (int j = 0; j < TN; j++) {
        accumulators[i][j] = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>(0.0);
      }
    }
  }

  [mutating]
  public void multiply(uint stage) {
    let base = stage * STAGE;
    [unroll]
    for (int step = 0; step < BK / (16 * SG_K); step++) {
      let depth = (uint(step * SG_K) + slice) * 2;
      CoopMat<BFloat16, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA> a[TM];
      CoopMat<BFloat16, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB> b[TN];
      [unroll]
      for (int i = 0; i < TM; i++) {
        a[i] = S.load<CoopMatMatrixUse.MatrixA, CoopMatMatrixLayout.RowMajor>(
          base + row_word + uint(16 * i) * STRIDE + depth, STRIDE
        );
      }
      [unroll]
      for (int j = 0; j < TN; j++) {
        b[j] = S.load<CoopMatMatrixUse.MatrixB, CoopMatMatrixLayout.ColumnMajor>(
          base + col_word + uint(16 * j) * STRIDE + depth, STRIDE
        );
      }
      [unroll]
      for (int i = 0; i < TM; i++) {
        [unroll]
        for (int j = 0; j < TN; j++) {
          accumulators[i][j] = coopMatMulAdd<float, false>(a[i], b[j], accumulators[i][j]);
        }
      }
    }
  }

  public void store(uint i, uint j, uint word) {
    S.store(accumulators[i][j], word, 4);
  }
}

uint4 or_component(uint4 word, uint index, uint bits) {
  return word | uint4(index == 0 ? bits : 0u, index == 1 ? bits : 0u, index == 2 ? bits : 0u, index == 3 ? bits : 0u);
}

struct TensorStager<B : IStorageFloat, S : ITensorShared, let BM : int, let BN : int, let BK : int, let THREADS : int> {
  static const uint STRIDE = uint(BK + 8) / 8;
  static const uint STAGE = uint(BM + BN) * STRIDE;
  static const uint A_COUNT = uint(BM * BK / 8);
  static const uint B_COUNT = uint(BN * BK / 8);
  static const uint Q_COUNT = uint(BN * BK / 16);
  static const int A_CHUNKS = (BM * BK / 8 + THREADS - 1) / THREADS;
  static const int B_CHUNKS = (BN * BK / 8 + THREADS - 1) / THREADS;
  static const int Q_CHUNKS = (BN * BK / 16 + THREADS - 1) / THREADS;

  Ptr<B, Access.Read> a;
  Ptr<B, Access.Read> dense;
  QuantizedWeights<B> weights;
  bool quantized;
  uint m;
  uint n;
  uint k;
  uint block_row;
  uint block_col;
  uint tid;
  bool a_vector;
  bool b_vector;

  uint4 a_words[A_CHUNKS];
  uint4 b_words[B_CHUNKS];
  float2 metadata[Q_CHUNKS];
  uint valid[Q_CHUNKS];

  [mutating]
  void load(uint k0) {
    [unroll]
    for (int i = 0; i < A_CHUNKS; i++) {
      let chunk = tid + uint(i * THREADS);
      var word = uint4(0);
      if (chunk < A_COUNT) {
        word = load_dense(a, block_row + chunk / uint(BK / 8), m, k0 + chunk % uint(BK / 8) * 8, a_vector);
      }
      a_words[i] = word;
    }
    if (!quantized) {
      [unroll]
      for (int i = 0; i < B_CHUNKS; i++) {
        let chunk = tid + uint(i * THREADS);
        var word = uint4(0);
        if (chunk < B_COUNT) {
          word = load_dense(dense, block_col + chunk / uint(BK / 8), n, k0 + chunk % uint(BK / 8) * 8, b_vector);
        }
        b_words[i] = word;
      }
      return;
    }
    [unroll]
    for (int i = 0; i < Q_CHUNKS; i++) {
      let chunk = tid + uint(i * THREADS);
      let row = block_col + chunk / uint(BK / 16);
      let column = k0 + chunk % uint(BK / 16) * 16;
      let count = chunk < Q_COUNT && row < n && column < k ? min(k - column, 16u) : 0u;
      var word = uint4(0);
      var scale = 0.0;
      var bias = 0.0;
      if (count > 0) {
        let bytes = weights.u4 ? (count + 1) / 2 : count;
        let source = weights.code_byte(row, column);
        if (b_vector && count == 16) {
          if (weights.u4) {
            word.xy = load_aligned<uint2, 8>(weights.codes, source);
          } else {
            word = load_aligned<uint4, 16>(weights.codes, source);
          }
        } else {
          [unroll]
          for (uint byte = 0; byte < 16; byte++) {
            if (byte < bytes) {
              word = or_component(word, byte / 4, uint(weights.codes[source + byte]) << (8 * (byte % 4)));
            }
          }
        }
        let group = weights.metadata(row, column / weights.group_size);
        scale = group.scale;
        bias = staged_bias<B>(group, weights.method);
      }
      b_words[i] = word;
      metadata[i] = float2(scale, bias);
      valid[i] = count;
    }
  }

  uint4 load_dense(Ptr<B, Access.Read> values, uint row, uint rows, uint column, bool vector) {
    var word = uint4(0);
    if (row < rows && column < k) {
      let source = row * k + column;
      if (vector && column + 8 <= k) {
        word = load_aligned<uint4, 16>(values, source);
      } else {
        let count = min(k - column, 8u);
        [unroll]
        for (uint value = 0; value < 8; value++) {
          if (value < count) {
            word = or_component(word, value / 2, values[source + value].storageBits() << (16 * (value % 2)));
          }
        }
      }
    }
    return word;
  }

  void store(uint stage) {
    let base = stage * STAGE;
    [unroll]
    for (int i = 0; i < A_CHUNKS; i++) {
      let chunk = tid + uint(i * THREADS);
      if (chunk < A_COUNT) {
        S.write(base + chunk / uint(BK / 8) * STRIDE + chunk % uint(BK / 8), a_words[i]);
      }
    }
    let b_base = base + uint(BM) * STRIDE;
    if (!quantized) {
      [unroll]
      for (int i = 0; i < B_CHUNKS; i++) {
        let chunk = tid + uint(i * THREADS);
        if (chunk < B_COUNT) {
          S.write(b_base + chunk / uint(BK / 8) * STRIDE + chunk % uint(BK / 8), b_words[i]);
        }
      }
      return;
    }
    [unroll]
    for (int i = 0; i < Q_CHUNKS; i++) {
      let chunk = tid + uint(i * THREADS);
      if (chunk < Q_COUNT) {
        uint4 staged[2] = { uint4(0), uint4(0) };
        if (valid[i] > 0) {
          let scale = metadata[i].x;
          let bias = metadata[i].y;
          uint bits[16];
          if (weights.u4) {
            let sixteenth = round_to<B>(scale / 16.0);
            [unroll]
            for (uint byte = 0; byte < 8; byte++) {
              let code = ((b_words[i][byte / 4] >> (8 * (byte % 4))) & 0xffu) ^ (weights.signed_codes ? 0x88u : 0u);
              bits[2 * byte] = staged_weight<B>(scale * float(code & 15u), bias).storageBits();
              bits[2 * byte + 1] = staged_weight<B>(sixteenth * float(code & 0xf0u), bias).storageBits();
            }
          } else if (weights.signed_codes) {
            let adjusted = round_to<B>(bias + round_to<B>(scale * 128.0));
            [unroll]
            for (uint byte = 0; byte < 16; byte++) {
              let code = int(b_words[i][byte / 4] << (24 - 8 * (byte % 4))) >> 24;
              bits[byte] = staged_weight<B>(scale * float(code), adjusted).storageBits();
            }
          } else {
            [unroll]
            for (uint byte = 0; byte < 16; byte++) {
              let code = (b_words[i][byte / 4] >> (8 * (byte % 4))) & 0xffu;
              bits[byte] = staged_weight<B>(scale * float(code), bias).storageBits();
            }
          }
          [unroll]
          for (uint value = 0; value < 16; value++) {
            if (value < valid[i]) {
              staged[value / 8][value / 2 % 4] |= bits[value] << (16 * (value % 2));
            }
          }
        }
        let word = b_base + chunk / uint(BK / 16) * STRIDE + chunk % uint(BK / 16) * 2;
        S.write(word, staged[0]);
        S.write(word + 1, staged[1]);
      }
    }
  }
}

public void gemm_tensor_run<
  B : IStorageFloat,
  D : IStorageFloat,
  S : ITensorShared,
  let BM : int,
  let BN : int,
  let BK : int,
  let SG_M : int,
  let SG_N : int,
  let SG_K : int
>(
  Ptr<B, Access.Read> a,
  Ptr<B, Access.Read> dense,
  QuantizedWeights<B> weights,
  bool quantized,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> output_bias,
  GemmParams params,
  uint2 tile,
  uint tid,
) {
  let TM = BM / (16 * SG_M);
  let TN = BN / (16 * SG_N);
  TensorStager<B, S, BM, BN, BK, SG_M * SG_N * SG_K * 32> stager;
  stager.a = a;
  stager.dense = dense;
  stager.weights = weights;
  stager.quantized = quantized;
  stager.m = params.M;
  stager.n = params.N;
  stager.k = params.K;
  stager.block_row = tile.y * uint(BM);
  stager.block_col = tile.x * uint(BN);
  stager.tid = tid;
  stager.a_vector = params.K % 8 == 0;
  stager.b_vector = quantized ? params.K % 16 == 0 : params.K % 8 == 0;

  var core = TensorCore<S, BM, BN, BK, SG_M, SG_N, SG_K>();
  let stages = (params.K + uint(BK) - 1) / uint(BK);
  if (stages > 0) {
    stager.load(0);
    stager.store(0);
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint stage = 0; stage < stages; stage++) {
    let next = stage + 1 < stages;
    if (next) {
      stager.load((stage + 1) * uint(BK));
    }
    core.multiply(stage & 1);
    if (next) {
      stager.store((stage + 1) & 1);
    }
    GroupMemoryBarrierWithGroupSync();
  }

  let subgroup = subgroup_id();
  let lane = WaveGetLaneIndex();
  let region = subgroup / uint(SG_K);
  let first_row = stager.block_row + region / uint(SG_N) * uint(TM * 16) + lane / 2;
  let first_col = stager.block_col + region % uint(SG_N) * uint(TN * 16) + lane % 2 * 8;
  [unroll]
  for (int i = 0; i < TM; i++) {
    [unroll]
    for (int j = 0; j < TN; j++) {
      GroupMemoryBarrierWithGroupSync();
      core.store(uint(i), uint(j), subgroup * 64);
      GroupMemoryBarrierWithGroupSync();
      let row = first_row + uint(16 * i);
      if (subgroup % uint(SG_K) == 0 && row < params.M) {
        float values[8];
        add_k_slices<S, SG_K>(subgroup * 64, lane, values);
        [unroll]
        for (uint value = 0; value < 8; value++) {
          let col = first_col + uint(16 * j) + value;
          if (col < params.N) {
            let bias = params.has_bias ? output_bias[col].toFloat() : 0.0;
            let output = gemm_output(values[value], bias, params);
            d[row * params.N + col] = D.fromFloat(output);
          }
        }
      }
    }
  }
}

public interface IInt8Shared<let MMA_K : int> : ISharedWords {
  static void write(uint word, uint4 value);
  static CoopMat<int8_t, MemoryScope.Subgroup, 16, MMA_K, CoopMatMatrixUse.MatrixA> load_a(uint word, uint stride);
  static CoopMat<int8_t, MemoryScope.Subgroup, MMA_K, 16, CoopMatMatrixUse.MatrixB> load_b(uint word, uint stride);
  static void store<T : ICoopElement>(
    CoopMat<T, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value,
    uint word,
    uint stride
  );
  static float fold(bool columns, uint index);
  static void stage_fold(bool columns, uint index, float value);
}

typealias Int8Dots = CoopMat<int, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
typealias Int8Sums = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

public struct Int8Gemm<
  let MMA_K : int,
  S : IInt8Shared<MMA_K>,
  B : IStorageFloat,
  D : IStorageFloat,
  let BM : int,
  let SG_M : int,
  let SG_N : int,
  let ELEMENT_FOLD : bool,
  let BK : int,
  let FOLD : int,
  let PREFETCH : int,
  let SG_K : int
> {
  static const int BN = SG_K > 1 ? 32 : 64;
  static const int PARTS = BK / FOLD;
  static const int THREADS = SG_M * SG_N * 32;
  static const int TM = BM / (16 * SG_M);
  static const int TN = BN / (16 * SG_N);
  static const uint STRIDE = uint(BK / 16 + 1);
  static const uint STAGE = uint(BM + BN) * STRIDE;
  static const uint SCRATCH = 2 * STAGE;
  static const int A_CHUNKS = (BM * BK / 16 + THREADS - 1) / THREADS;
  static const int B_RUNS = (BN * BK / 32 + THREADS - 1) / THREADS;
  static const int B_CHUNKS = 2 * B_RUNS;
  static const int M_CHUNKS = (PARTS * (BM + BN) + THREADS - 1) / THREADS;

  public static void run(
    Ptr<int8_t, Access.Read> a_codes,
    Ptr<float, Access.Read> a_scales,
    Ptr<int, Access.Read> group_sums,
    QuantizedWeights<B> weights,
    Ptr<D, Access.ReadWrite> d,
    Ptr<B, Access.Read> output_bias,
    Ptr<uint, Access.ReadWrite> partials,
    GemmParams params,
    uint2 tile,
    uint invocation,
    uint split_phase,
    uint splits,
    uint split_index,
    Ptr<int, Access.Read> rht_factors,
    bool has_rht,
    bool rht_bias,
  ) {
    static_assert(SG_K == 1 || (SG_M * SG_N == 1 && PREFETCH > 1), "a K slice is one ring subgroup");
    let m = params.M;
    let n = params.N;
    let k = params.K;
    let block_row = tile.y * uint(BM);
    let block_col = tile.x * uint(BN);
    let u4 = weights.u4;
    let scale_bias = weights.method == QuantizationMethod.ScaleBias;
    let subgroup = subgroup_id() / uint(SG_K);
    let slice = subgroup_id() % uint(SG_K);
    let lane = WaveGetLaneIndex();
    let tid = SG_K > 1 ? lane : invocation;
    let slice_word = slice * 2 * STAGE;
    let slice_slot = slice * uint(2 * PARTS);
    let row_word = (subgroup / uint(SG_N)) * uint(TM * 16) * STRIDE;
    let col_word = uint(BM) * STRIDE + (subgroup % uint(SG_N)) * uint(TN * 16) * STRIDE;

    uint4 a_chunks[PREFETCH][A_CHUNKS] = {};
    uint4 b_chunks[PREFETCH][B_CHUNKS] = {};
    float2 metadata_chunks[PREFETCH][M_CHUNKS] = {};
    B column_scales[PREFETCH][M_CHUNKS] = {};
    B column_biases[PREFETCH][M_CHUNKS] = {};
    let row_base = (subgroup / uint(SG_N)) * uint(TM * 16);
    let col_base = (subgroup % uint(SG_N)) * uint(TN * 16);

    Int8Dots dots[TM][TN];
    Int8Sums folded[TM][TN];
    float sums[TM][TN][8];
    [unroll]
    for (int i = 0; i < TM; i++) {
      [unroll]
      for (int j = 0; j < TN; j++) {
        dots[i][j] = Int8Dots(0);
        folded[i][j] = Int8Sums(0.0);
        [unroll]
        for (uint value = 0; value < 8; value++) {
          sums[i][j][value] = 0.0;
        }
      }
    }

    let k_first = split_index * (k / splits) + slice * (k / splits / uint(SG_K));
    let stages = split_phase == 2 ? 0 : k / splits / uint(SG_K) / uint(BK);
    let peeled = ELEMENT_FOLD && !u4 && BM == 64;
    let steps = (stages + uint(2 * PREFETCH - 1)) / uint(PREFETCH) * uint(PREFETCH);
    [unroll]
    for (int phase = 0; phase < 3; phase++) {
      let begin = phase == 0 ? 0u : phase == 1 ? 1u : stages;
      let end = phase == 0 ? min(stages, 1u) : phase == 1 ? stages : stages + 1;
      for (uint round = begin; peeled ? round < end : phase == 0 && round < steps; round += uint(PREFETCH)) {
        [unroll]
        for (int slot = 0; slot < PREFETCH; slot++) {
          let stage = round + uint(slot);
          if (PREFETCH > 1 && stage < stages) {
            static_assert(
              A_CHUNKS * THREADS == BM * BK / 16 && B_RUNS * THREADS == BN * BK / 32, "whole stages a step"
            );
            let k0 = k_first + stage * uint(BK);
            [unroll]
            for (int i = 0; i < A_CHUNKS; i++) {
              let chunk = tid + uint(i * THREADS);
              let row = min(block_row + chunk / uint(BK / 16), m - 1);
              a_chunks[slot][i] = load_aligned<uint4, 16>(a_codes, row * k + k0 + chunk % uint(BK / 16) * 16);
            }
            [unroll]
            for (int i = 0; i < B_RUNS; i++) {
              let run = tid + uint(i * THREADS);
              let row = min(block_col + run / uint(BK / 32), n - 1);
              let column = k0 + run % uint(BK / 32) * 32;
              let zero =
                row * weights.zero_point_output_stride + column / weights.group_size * weights.zero_point_group_stride;
              let zero_byte =
                weights.method == QuantizationMethod.ScaleZeroPoint ? uint(weights.zero_points[zero / 2]) : 0u;
              b_chunks[slot][2 * i] = load_aligned<uint4, 16>(weights.codes, weights.code_byte(row, column));
              b_chunks[slot][2 * i + 1] = uint4(zero_byte, zero % 2, 0, 0);
            }
            [unroll]
            for (int i = 0; i < M_CHUNKS; i++) {
              let entry = min(tid + uint(i * THREADS), uint(PARTS * (BM + BN)) - 1);
              let part = entry / uint(BM + BN);
              let index = entry % uint(BM + BN);
              let end = k0 + uint(FOLD) * (part + 1);
              let group = (end - 1) / weights.group_size;
              let input = min(block_row + min(index, uint(BM) - 1), m - 1);
              let column = min(block_col + (index < uint(BM) ? 0u : index - uint(BM)), n - 1);
              let scale_index = column * weights.scale_output_stride + group * weights.scale_group_stride;
              let scale_group = (end - 1) / ACTIVATION_QUANTIZATION_TILE;
              metadata_chunks[slot][i] = float2(
                a_scales[input * (k / ACTIVATION_QUANTIZATION_TILE) + scale_group],
                scale_bias ? asfloat(group_sums[input * (k / weights.group_size) + group]) : 0.0
              );
              column_scales[slot][i] = weights.scales[scale_index];
              if (scale_bias) {
                column_biases[slot][i] = weights.biases[scale_index];
              }
            }
          } else if (PREFETCH == 1 && (peeled ? phase < 2 : stage < stages)) {
            let k0 = k_first + stage * uint(BK);
            [unroll]
            for (int i = 0; i < A_CHUNKS; i++) {
              let chunk = tid + uint(i * THREADS);
              let row = block_row + chunk / uint(BK / 16);
              var word = uint4(0);
              if (chunk < uint(BM * BK / 16) && row < m) {
                word = load_aligned<uint4, 16>(a_codes, row * k + k0 + chunk % uint(BK / 16) * 16);
              }
              a_chunks[slot][i] = word;
            }
            [unroll]
            for (int i = 0; i < B_RUNS; i++) {
              let run = tid + uint(i * THREADS);
              let row = block_col + run / uint(BK / 32);
              let column = k0 + run % uint(BK / 32) * 32;
              uint4 words[2] = { uint4(0), uint4(0) };
              if (run < uint(BN * BK / 32) && row < n) {
                let metadata = weights.metadata(row, column / weights.group_size);
                let source = weights.code_byte(row, column);
                uint4 packed[2];
                [unroll]
                for (uint half = 0; half < 2; half++) {
                  packed[half] = uint4(0);
                  if (half == 0 || !u4) {
                    packed[half] = load_aligned<uint4, 16>(weights.codes, source + 16 * half);
                  }
                }
                [unroll]
                for (uint half = 0; half < 2; half++) {
                  if (u4) {
                    let low = centered_u4(packed[0][2 * half], metadata.zero_point, weights.signed_codes);
                    let high = centered_u4(packed[0][2 * half + 1], metadata.zero_point, weights.signed_codes);
                    words[half] = uint4(low.x, low.y, high.x, high.y);
                  } else {
                    [unroll]
                    for (uint component = 0; component < 4; component++) {
                      words[half][component] = centered_u8(packed[half][component], weights.signed_codes);
                    }
                  }
                }
              }
              b_chunks[slot][2 * i] = words[0];
              b_chunks[slot][2 * i + 1] = words[1];
            }
            if (ELEMENT_FOLD) {
              [unroll]
              for (int i = 0; i < M_CHUNKS; i++) {
                let entry = tid + uint(i * THREADS);
                let part = entry / uint(BM + BN);
                let index = entry % uint(BM + BN);
                let end = k0 + uint(FOLD) * (part + 1);
                var value = float2(0.0);
                let group_end = (stage * uint(BK) + uint(FOLD) * (part + 1)) % weights.group_size == 0;
                if (entry < uint(PARTS * (BM + BN)) && group_end) {
                  let group = (end - 1) / weights.group_size;
                  if (index < uint(BM)) {
                    let input = min(block_row + index, m - 1);
                    float a_scale;
                    if (BM == 64 && BK == 64 && weights.group_size == 64 && (stage & 1u) != 0) {
                      a_scale = S.fold(false, 128 + index);
                    } else {
                      a_scale = a_scales[
                        input * (k / ACTIVATION_QUANTIZATION_TILE) + (end - 1) / ACTIVATION_QUANTIZATION_TILE
                      ];
                    }
                    let group_sum = scale_bias ? group_sums[input * (k / weights.group_size) + group] : 0;
                    value = float2(a_scale, a_scale * float(group_sum));
                  } else {
                    let metadata = weights.metadata(min(block_col + index - uint(BM), n - 1), group);
                    value = float2(metadata.scale, metadata.bias);
                  }
                }
                metadata_chunks[slot][i] = value;
              }
            }
          }
          let multiplies = stage >= uint(PREFETCH) && stage - uint(PREFETCH) < stages;
          if (peeled ? (phase == 1 || phase == 2 && stage > 0) : multiplies) {
            let multiplied = stage - uint(PREFETCH);
            let base = slice_word + (multiplied & 1) * STAGE;
            [unroll]
            for (int step = 0; step < BK / MMA_K; step++) {
              CoopMat<int8_t, MemoryScope.Subgroup, 16, MMA_K, CoopMatMatrixUse.MatrixA> a[TM];
              [unroll]
              for (int i = 0; i < TM; i++) {
                a[i] = S.load_a(base + row_word + uint(16 * i) * STRIDE + uint(MMA_K / 16 * step), STRIDE);
              }
              [unroll]
              for (int j = 0; j < TN; j++) {
                let b = S.load_b(base + col_word + uint(16 * j) * STRIDE + uint(MMA_K / 16 * step), STRIDE);
                [unroll]
                for (int i = 0; i < TM; i++) {
                  dots[i][j] = coopMatMulAdd<int, false>(a[i], b, dots[i][j]);
                }
              }
              let offset = multiplied * uint(BK) + uint(MMA_K * (step + 1));
              let end = k_first + offset;
              if (ELEMENT_FOLD && offset % weights.group_size == 0) {
                let fold_slot = slice_slot + (multiplied & 1) * uint(PARTS) + uint(MMA_K * (step + 1) / FOLD - 1);
                [unroll]
                for (int i = 0; i < TM; i++) {
                  [unroll]
                  for (int j = 0; j < TN; j++) {
                    let rows = fold_slot * 128 + row_base + uint(16 * i);
                    let cols = fold_slot * 128 + col_base + uint(16 * j);
                    folded[i][j] = makeTuple(folded[i][j], Int8Sums(dots[i][j])).MapElement(
                      (uint32_t row, uint32_t column, float sum, float dot) => {
                        let value = fma(S.fold(false, rows + row) * S.fold(true, cols + column), dot, sum);
                        return scale_bias ? fma(S.fold(true, cols + 64 + column), S.fold(false, rows + 64 + row), value) : value;
                      });
                    dots[i][j] = Int8Dots(0);
                  }
                }
              } else if (!ELEMENT_FOLD && offset % weights.group_size == 0) {
                let group = (end - 1) / weights.group_size;
                let scale_word = SCRATCH + subgroup * uint(TM * TN) * 64;
                [unroll]
                for (int i = 0; i < TM; i++) {
                  [unroll]
                  for (int j = 0; j < TN; j++) {
                    S.store(dots[i][j], scale_word + uint(i * TN + j) * 64, 4);
                    dots[i][j] = Int8Dots(0);
                  }
                }
                GroupMemoryBarrierWithGroupSync();
                [unroll]
                for (int i = 0; i < TM; i++) {
                  let row = block_row + (subgroup / uint(SG_N)) * uint(TM * 16) + uint(16 * i) + lane / 2;
                  let input = min(row, m - 1);
                  let a_scale = a_scales[
                    input * (k / ACTIVATION_QUANTIZATION_TILE) + (end - 1) / ACTIVATION_QUANTIZATION_TILE
                  ];
                  let group_sum = scale_bias ? group_sums[input * (k / weights.group_size) + group] : 0;
                  [unroll]
                  for (int j = 0; j < TN; j++) {
                    let word = scale_word + uint(i * TN + j) * 64 + lane * 2;
                    let low = S.read(word);
                    let high = S.read(word + 1);
                    [unroll]
                    for (uint value = 0; value < 8; value++) {
                      let col =
                        block_col + (subgroup % uint(SG_N)) * uint(TN * 16) + uint(16 * j) + lane % 2 * 8 + value;
                      let metadata = weights.metadata(min(col, n - 1), group);
                      let dot = int(value < 4 ? low[value] : high[value - 4]);
                      sums[i][j][value] = int8_fold(sums[i][j][value], a_scale, metadata, dot, scale_bias, group_sum);
                    }
                  }
                }
                GroupMemoryBarrierWithGroupSync();
              }
            }
          }
          let staged = stage + 1 - uint(PREFETCH);
          let ring = (slot + 1) % PREFETCH;
          if (peeled ? phase < 2 : staged < stages) {
            let base = slice_word + (staged & 1) * STAGE;
            [unroll]
            for (int i = 0; i < A_CHUNKS; i++) {
              let chunk = tid + uint(i * THREADS);
              if (chunk < uint(BM * BK / 16)) {
                S.write(base + chunk / uint(BK / 16) * STRIDE + chunk % uint(BK / 16), a_chunks[ring][i]);
              }
            }
            [unroll]
            for (int i = 0; i < B_RUNS; i++) {
              let run = tid + uint(i * THREADS);
              if (run < uint(BN * BK / 32)) {
                let word = base + uint(BM) * STRIDE + run / uint(BK / 32) * STRIDE + run % uint(BK / 32) * 2;
                if (PREFETCH > 1) {
                  let zero_word = b_chunks[ring][2 * i + 1];
                  let stored = (zero_word.x >> (4 * zero_word.y)) & 15u;
                  let zero = weights.method == QuantizationMethod.ScaleZeroPoint ? stored
                    : weights.method == QuantizationMethod.ScaleBias ? 0u : 8u;
                  [unroll]
                  for (uint half = 0; half < 2; half++) {
                    let low = centered_u4(b_chunks[ring][2 * i][2 * half], zero, weights.signed_codes);
                    let high = centered_u4(b_chunks[ring][2 * i][2 * half + 1], zero, weights.signed_codes);
                    S.write(word + half, uint4(low.x, low.y, high.x, high.y));
                  }
                } else {
                  S.write(word, b_chunks[ring][2 * i]);
                  S.write(word + 1, b_chunks[ring][2 * i + 1]);
                }
              }
            }
            if (ELEMENT_FOLD) {
              [unroll]
              for (int i = 0; i < M_CHUNKS; i++) {
                let entry = tid + uint(i * THREADS);
                let index = entry % uint(BM + BN);
                let fold_slot = slice_slot + (staged & 1) * uint(PARTS) + entry / uint(BM + BN);
                if (entry < uint(PARTS * (BM + BN))) {
                  if (index < uint(BM)) {
                    let row_entry = metadata_chunks[ring][i];
                    S.stage_fold(false, fold_slot * 128 + index, row_entry.x);
                    S.stage_fold(
                      false, fold_slot * 128 + 64 + index,
                      PREFETCH > 1 ? row_entry.x * float(asint(row_entry.y)) : row_entry.y
                    );
                  } else {
                    S.stage_fold(
                      true, fold_slot * 128 + index - uint(BM),
                      PREFETCH > 1 ? column_scales[ring][i].toFloat() : metadata_chunks[ring][i].x
                    );
                    S.stage_fold(
                      true, fold_slot * 128 + 64 + index - uint(BM),
                      PREFETCH == 1 ? metadata_chunks[ring][i].y : scale_bias ? column_biases[ring][i].toFloat() : 0.0
                    );
                  }
                }
              }
            }
          }
          if (SG_K > 1) {
            GroupMemoryBarrierWithWaveSync();
          } else {
            GroupMemoryBarrierWithGroupSync();
          }
        }
      }
    }
    if (ELEMENT_FOLD && split_phase != 2) {
      [unroll]
      for (int i = 0; i < TM; i++) {
        [unroll]
        for (int j = 0; j < TN; j++) {
          GroupMemoryBarrierWithGroupSync();
          S.store(folded[i][j], subgroup_id() * 64, 4);
          GroupMemoryBarrierWithGroupSync();
          add_k_slices<S, SG_K>(subgroup * uint(SG_K) * 64, lane, sums[i][j]);
        }
      }
    }

    let transform = has_rht && split_phase != 1;
    if (transform) {
      GroupMemoryBarrierWithGroupSync();
    }
    [unroll]
    for (int i = 0; i < TM; i++) {
      let row_local = (subgroup / uint(SG_N)) * uint(TM * 16) + uint(16 * i) + lane / 2;
      let row = block_row + row_local;
      [unroll]
      for (int j = 0; j < TN; j++) {
        let col_local = (subgroup % uint(SG_N)) * uint(TN * 16) + uint(16 * j) + lane % 2 * 8;
        let owned = uint((i * TN + j) % SG_K) == slice;
        uint staged[8];
        [unroll]
        for (uint value = 0; value < 8; value++) {
          let col = block_col + col_local + value;
          staged[value] = 0;
          if (owned && row < m && col < n && split_phase == 1) {
            partials[(split_index * m + row) * n + col] = asuint(sums[i][j][value]);
          } else if (owned && row < m && col < n) {
            var sum = sums[i][j][value];
            if (split_phase == 2) {
              sum = asfloat(partials[row * n + col]);
              for (uint split = 1; split < splits; split++) {
                sum += asfloat(partials[(split * m + row) * n + col]);
              }
            }
            let bias = params.has_bias ? output_bias[col].toFloat() : 0.0;
            let output = D.fromFloat(gemm_output(sum, bias, params));
            staged[value] = asuint(output.toFloat());
            if (!transform) {
              d[row * n + col] = output;
            }
          }
        }
        if (transform && owned) {
          let word = row_local * uint(BN / 4) + col_local / 4;
          S.write(word, uint4(staged[0], staged[1], staged[2], staged[3]));
          S.write(word + 1, uint4(staged[4], staged[5], staged[6], staged[7]));
        }
      }
    }
    if (transform) {
      GroupMemoryBarrierWithGroupSync();
      static const int BLOCKS = BN / 32;
      static const int SUBGROUPS = SG_M * SG_N * SG_K;
      static_assert(SUBGROUPS % BLOCKS == 0, "every subgroup keeps one column block");
      let col_local = subgroup_id() % uint(BLOCKS) * 32 + lane;
      let col = block_col + col_local;
      if (col < n) {
        let factor = rht_factors[col];
        let bias = rht_bias ? output_bias[col].toFloat() : 0.0;
        [loop]
        for (int step = 0; step < BM * BLOCKS / SUBGROUPS; step++) {
          let row_local = (uint(step * SUBGROUPS) + subgroup_id()) / uint(BLOCKS);
          let row = block_row + row_local;
          if (row < m) {
            let value = asfloat(S.read(row_local * uint(BN / 4) + col_local / 4)[col_local % 4]);
            var output = hadamard_butterflies(value, HADAMARD_TRANSFORM_BLOCK_SIZE) / sqrt(float(HADAMARD_TRANSFORM_BLOCK_SIZE)) * float(factor);
            if (rht_bias) {
              output = D.fromFloat(output).toFloat() + bias;
            }
            d[row * n + col] = D.fromFloat(output);
          }
        }
      }
    }
  }
}
