import definitions;
import common.hadamard;
import matmul.common.matmul;
import common.quantized_value;
import generated.gemm;
import generated.quantization_method;
import matmul.common.int8_operands;
import matmul.common.loader;
import matmul.gemm.common.simdgroup_mma_core;

using namespace linalg;

// The cooperative-matrix Gemm core: SG_M x SG_N x SG_K subgroups compute a BM x BN block from 16-bit A and B staged in
// two BK-deep stages, rows padded by 8 values; each subgroup owns (BM / SG_M) x (BN / SG_N) as 16 x 16 FP32
// accumulators over every SG_K-th 16-deep step, and the K slices add in slice order. Quantized B stages rounded to N.

// A workgroup's staging as 16-byte words of 8 16-bit values, which cooperative-matrix loads index and stride in words.
public interface ITensorShared<N : ICoopElement> {
  static uint4 read(uint word);
  static void write(uint word, uint4 value);
  static CoopMat<N, MemoryScope.Subgroup, 16, 16, USE> load<
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
  N : ICoopElement,
  S : ITensorShared<N>,
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
      CoopMat<N, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA> a[TM];
      CoopMat<N, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB> b[TN];
      [unroll]
      for (int i = 0; i < TM; i++) {
        a[i] = S.load<CoopMatMatrixUse.MatrixA, CoopMatMatrixLayout.RowMajor>(
          base + row_word + uint(16 * i) * STRIDE + depth, STRIDE
        );
      }
      // B is staged as its columns of K: column major.
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

  // Accumulator (i, j) as 16 rows of 16 FP32 values, 4 words a row, from `word`.
  public void store(uint i, uint j, uint word) {
    S.store(accumulators[i][j], word, 4);
  }
}

// `word` with `bits` or-ed into component `index`, selected per component rather than indexed.
uint4 or_component(uint4 word, uint index, uint bits) {
  return word | uint4(index == 0 ? bits : 0u, index == 1 ? bits : 0u, index == 2 ? bits : 0u, index == 3 ? bits : 0u);
}

// One workgroup's staging of A and B. Chunks of a stage are 8 values of A or full-precision B, or 16 quantized values
// of B in one group, chunk c of a stage belonging to invocation c mod THREADS. Values past M, N or K stage as +0.
struct TensorStager<B : IModelFloat16, N : ICoopElement, S : ITensorShared<N>, let BM : int, let BN : int, let BK : int, let THREADS : int> {
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
  // Per quantized chunk: scale, bias and the count of values below K.
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
        let source = row * (weights.u4 ? k / 2 : k) + (weights.u4 ? column / 2 : column);
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
        bias = weights.method == QuantizationMethod.ScaleBias ? group.bias : round_to<B>(-scale * float(group.zero_point));
      }
      b_words[i] = word;
      metadata[i] = float2(scale, bias);
      valid[i] = count;
    }
  }

  // Eight values of `row` from `column`, +0 past `rows` or k, as one 16-byte load where k is a multiple of 8.
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
            // A high nibble is read in place, 16 times its code, against the scale over 16.
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

// The block at `tile` (column tile, row tile), staging `dense` or, when `quantized`, `weights`. Every invocation of the
// workgroup must call it.
public void gemm_tensor_run<
  B : IModelFloat16,
  N : ICoopElement,
  D : IModelFloat,
  S : ITensorShared<N>,
  let BM : int,
  let BN : int,
  let BK : int,
  let SG_M : int,
  let SG_N : int,
  let SG_K : int
>(
  DenseOperand<B> left,
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
  TensorStager<B, N, S, BM, BN, BK, SG_M * SG_N * SG_K * 32> stager;
  stager.a = left.values;
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

  var core = TensorCore<N, S, BM, BN, BK, SG_M, SG_N, SG_K>();
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

  // Each accumulator through threadgroup memory, which the stages no longer use: the region's K slices summed in slice
  // order by the first slice's lanes, 8 adjacent outputs of one row each.
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
        [unroll]
        for (uint value = 0; value < 8; value++) {
          values[value] = 0.0;
        }
        [unroll]
        for (int split = 0; split < SG_K; split++) {
          let word = (subgroup + uint(split)) * 64 + lane * 2;
          let low = S.read(word);
          let high = S.read(word + 1);
          [unroll]
          for (uint value = 0; value < 4; value++) {
            values[value] = split == 0 ? asfloat(low[value]) : values[value] + asfloat(low[value]);
            values[4 + value] = split == 0 ? asfloat(high[value]) : values[4 + value] + asfloat(high[value]);
          }
        }
        [unroll]
        for (uint value = 0; value < 8; value++) {
          let col = first_col + uint(16 * j) + value;
          if (col < params.N) {
            let bias = params.has_bias ? output_bias[col].toFloat() : 0.0;
            let output = gemm_output(values[value], bias, params);
            d[row * params.leading_dimension_d + col] = D.fromFloat(output);
          }
        }
      }
    }
  }
}

// The INT8 cooperative-matrix core: SG_M x SG_N subgroups compute a BM x 64 block from 64-deep stages of activation
// codes and weight codes centered to signed bytes, rows padded by 16 bytes, as 16 x 16 INT32 accumulators exact within a
// weight group. At each group's end every lane folds its outputs into FP32 as int8_fold does and the accumulators restart.
public interface IInt8Shared<let MMA_K : int> {
  static uint4 read(uint word);
  static void write(uint word, uint4 value);
  static CoopMat<int8_t, MemoryScope.Subgroup, 16, MMA_K, CoopMatMatrixUse.MatrixA> load_a(uint word, uint stride);
  static CoopMat<int8_t, MemoryScope.Subgroup, MMA_K, 16, CoopMatMatrixUse.MatrixB> load_b(uint word, uint stride);
  static void store(
    CoopMat<int, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value,
    uint word,
    uint stride
  );
  static void store_float(
    CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value,
    uint word,
    uint stride
  );
}

// Fold metadata [slot][kind][index] of each 32-deep step ending a weight group, slot = stage parity * 2 + step: rows
// hold the activation scale and its ScaleBias correction, columns the weight scale and bias.
groupshared float int8_fold_rows[INT8_FOLD_ENTRIES + INT8_FOLD_PADDING];
groupshared float int8_fold_cols[INT8_FOLD_ENTRIES];

typealias Int8Dots = CoopMat<int, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
typealias Int8Sums = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

// `sums` plus int8_fold of `dots` per element (VK_NV_cooperative_matrix2), from the metadata staged in `slot` for the
// accumulator at (`row_base`, `col_base`). The dots convert to FP32 first, exactly below 2^24.
Int8Sums int8_element_fold(Int8Sums sums, Int8Dots dots, uint slot, uint row_base, uint col_base, bool scale_bias) {
  let rows = slot * 128 + row_base;
  let cols = slot * 128 + col_base;
  return makeTuple(sums, Int8Sums(dots)).MapElement((uint32_t row, uint32_t column, float sum, float dot) => {
    let value = fma(int8_fold_rows[rows + row] * int8_fold_cols[cols + column], dot, sum);
    return scale_bias ? fma(int8_fold_cols[cols + 64 + column], int8_fold_rows[rows + 64 + row], value) : value;
  });
}

public struct Int8Gemm<
  let MMA_K : int,
  S : IInt8Shared<MMA_K>,
  B : IModelFloat,
  D : IModelFloat,
  let BM : int,
  let SG_M : int,
  let SG_N : int,
  let ELEMENT_FOLD : bool,
  let BK : int
> {
  static const int BN = 64;
  // A stage's two halves, each folding the weight groups that end there.
  static const int HALF = BK / 2;
  static const int THREADS = SG_M * SG_N * 32;
  static const int TM = BM / (16 * SG_M);
  static const int TN = BN / (16 * SG_N);
  // Rows of BK bytes padded by one word.
  static const uint STRIDE = uint(BK / 16 + 1);
  static const uint STAGE = uint(BM + BN) * STRIDE;
  static const uint SCRATCH = 2 * STAGE;
  static const int A_CHUNKS = (BM * BK / 16 + THREADS - 1) / THREADS;
  static const int B_RUNS = (BN * BK / 32 + THREADS - 1) / THREADS;
  static const int B_CHUNKS = 2 * B_RUNS;
  // A stage's fold metadata entries, BM rows then BN columns for each of its halves.
  static const int M_CHUNKS = (2 * (BM + BN) + THREADS - 1) / THREADS;

  // Split K (split_phase 1 and 2, `splits` of 2 to 8 whole 128-value runs each): phase 1 stores split split_index's FP32
  // sums as bits in `partials` [splits][m][n]; phase 2 adds them in split order and finishes as phase 0 does.
  // `has_rht` finishes with ActivationTransform's output transform of each row's 32-column blocks, then `output_bias`
  // where `rht_bias`: the outputs, rounded as stored, pass through the stages' memory to the subgroups.
  public static void run(
    Int8Activations activations,
    Ptr<int, Access.Read> group_sums,
    QuantizedWeights<B> weights,
    Ptr<D, Access.ReadWrite> d,
    Ptr<B, Access.Read> output_bias,
    Ptr<uint, Access.ReadWrite> partials,
    GemmParams params,
    uint2 tile,
    uint tid,
    uint split_phase,
    uint splits,
    uint split_index,
    Ptr<int, Access.Read> rht_factors,
    bool has_rht,
    bool rht_bias,
  ) {
    let m = params.M;
    let n = params.N;
    let k = params.K;
    let block_row = tile.y * uint(BM);
    let block_col = tile.x * uint(BN);
    let u4 = weights.u4;
    let scale_bias = weights.method == QuantizationMethod.ScaleBias;
    let subgroup = subgroup_id();
    let lane = WaveGetLaneIndex();
    let row_word = (subgroup / uint(SG_N)) * uint(TM * 16) * STRIDE;
    let col_word = uint(BM) * STRIDE + (subgroup % uint(SG_N)) * uint(TN * 16) * STRIDE;

    // The next stage's chunks of 16 values: activation codes as stored, weight codes centered.
    uint4 a_chunks[A_CHUNKS] = {};
    uint4 b_chunks[B_CHUNKS] = {};
    // Per-element folds: the next stage's metadata entries.
    float2 metadata_chunks[M_CHUNKS] = {};
    let row_base = (subgroup / uint(SG_N)) * uint(TM * 16);
    let col_base = (subgroup % uint(SG_N)) * uint(TN * 16);

    Int8Dots dots[TM][TN];
    // Per-element folds accumulate in `folded` across K; lane folds in `sums`, which both finish from.
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

    // Each split starts on a 128-value boundary, hence on every supported weight-group boundary.
    let k_first = split_index * (k / splits);
    let stages = split_phase == 2 ? 0 : k / splits / uint(BK);
    // Peeled into prologue (stage 0 loads), steady state (stage loads while stage - 1 multiplies) and epilogue (the last
    // stage multiplies), each the same sequence with constant conditions.
    [unroll]
    for (int phase = 0; phase < 3; phase++) {
      let begin = phase == 0 ? 0u : phase == 1 ? 1u : stages;
      let end = phase == 0 ? min(stages, 1u) : phase == 1 ? stages : stages + 1;
      for (uint stage = begin; (ELEMENT_FOLD && !u4 && BM == 64) ? stage < end : phase == 0 && stage <= stages; stage++) {
        // Stage `stage` loads while stage - 1 multiplies.
        if ((ELEMENT_FOLD && !u4 && BM == 64) ? phase < 2 : stage < stages) {
          let k0 = k_first + stage * uint(BK);
          [unroll]
          for (int i = 0; i < A_CHUNKS; i++) {
            let chunk = tid + uint(i * THREADS);
            let row = block_row + chunk / uint(BK / 16);
            var word = uint4(0);
            if (chunk < uint(BM * BK / 16) && row < m) {
              word = load_aligned<uint4, 16>(activations.codes, row * k + k0 + chunk % uint(BK / 16) * 16);
            }
            a_chunks[i] = word;
          }
          // One run of 32 weights of one row and group per invocation: one 16-byte W4 load or two W8 loads, contiguous
          // across a subgroup in the Tiled layout, then two staged words.
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
            b_chunks[2 * i] = words[0];
            b_chunks[2 * i + 1] = words[1];
          }
          if (ELEMENT_FOLD) {
            // The metadata the lane fold reads for the steps that end a weight group, rows and columns clamped alike.
            [unroll]
            for (int i = 0; i < M_CHUNKS; i++) {
              let entry = tid + uint(i * THREADS);
              let part = entry / uint(BM + BN);
              let index = entry % uint(BM + BN);
              let end = k0 + uint(HALF) * (part + 1);
              var value = float2(0.0);
              if (entry < uint(2 * (BM + BN)) && (stage * uint(BK) + uint(HALF) * (part + 1)) % weights.group_size == 0) {
                let group = (end - 1) / weights.group_size;
                if (index < uint(BM)) {
                  let input = min(block_row + index, m - 1);
                  float a_scale;
                  if (BM == 64 && BK == 64 && weights.group_size == 64 && (stage & 1u) != 0) {
                    a_scale = int8_fold_rows[128 + index];
                  } else {
                    a_scale = activations.scales[input * (k / INT8_SCALE_GROUP) + (end - 1) / INT8_SCALE_GROUP];
                  }
                  let group_sum = scale_bias ? group_sums[input * (k / weights.group_size) + group] : 0;
                  value = float2(a_scale, a_scale * float(group_sum));
                } else {
                  let metadata = weights.metadata(min(block_col + index - uint(BM), n - 1), group);
                  value = float2(metadata.scale, metadata.bias);
                }
              }
              metadata_chunks[i] = value;
            }
          }
        }
        if ((ELEMENT_FOLD && !u4 && BM == 64) ? (phase == 1 || phase == 2 && stage > 0) : stage > 0) {
          let base = ((stage - 1) & 1) * STAGE;
          [unroll]
          for (int step = 0; step < BK / MMA_K; step++) {
            CoopMat<int8_t, MemoryScope.Subgroup, 16, MMA_K, CoopMatMatrixUse.MatrixA> a[TM];
            [unroll]
            for (int i = 0; i < TM; i++) {
              a[i] = S.load_a(base + row_word + uint(16 * i) * STRIDE + uint(MMA_K / 16 * step), STRIDE);
            }
            [unroll]
            for (int j = 0; j < TN; j++) {
              // Weight rows are staged as columns of K: column major.
              let b = S.load_b(base + col_word + uint(16 * j) * STRIDE + uint(MMA_K / 16 * step), STRIDE);
              [unroll]
              for (int i = 0; i < TM; i++) {
                dots[i][j] = coopMatMulAdd<int, false>(a[i], b, dots[i][j]);
              }
            }
            // A weight group ends: fold the accumulators and restart them.
            let offset = (stage - 1) * uint(BK) + uint(MMA_K * (step + 1));
            let end = k_first + offset;
            if (ELEMENT_FOLD && offset % weights.group_size == 0) {
              // In registers, from the metadata staged with this stage.
              let slot = ((stage - 1) & 1) * 2 + uint(MMA_K * (step + 1) / HALF - 1);
              [unroll]
              for (int i = 0; i < TM; i++) {
                [unroll]
                for (int j = 0; j < TN; j++) {
                  folded[i][j] = int8_element_fold(
                    folded[i][j], dots[i][j], slot, row_base + uint(16 * i), col_base + uint(16 * j), scale_bias
                  );
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
                let a_scale = activations.scales[input * (k / INT8_SCALE_GROUP) + (end - 1) / INT8_SCALE_GROUP];
                let group_sum = scale_bias ? group_sums[input * (k / weights.group_size) + group] : 0;
                [unroll]
                for (int j = 0; j < TN; j++) {
                  let word = scale_word + uint(i * TN + j) * 64 + lane * 2;
                  let low = S.read(word);
                  let high = S.read(word + 1);
                  [unroll]
                  for (uint value = 0; value < 8; value++) {
                    let col = block_col + (subgroup % uint(SG_N)) * uint(TN * 16) + uint(16 * j) + lane % 2 * 8 + value;
                    let metadata = weights.metadata(min(col, n - 1), group);
                    let dot = int(value < 4 ? low[value] : high[value - 4]);
                    sums[i][j][value] = int8_fold(sums[i][j][value], a_scale, metadata, dot, scale_bias, group_sum);
                  }
                }
              }
              // Every lane has read the scratch before the next group's accumulators overwrite it.
              GroupMemoryBarrierWithGroupSync();
            }
          }
        }
        if ((ELEMENT_FOLD && !u4 && BM == 64) ? phase < 2 : stage < stages) {
          let base = (stage & 1) * STAGE;
          [unroll]
          for (int i = 0; i < A_CHUNKS; i++) {
            let chunk = tid + uint(i * THREADS);
            if (chunk < uint(BM * BK / 16)) {
              S.write(base + chunk / uint(BK / 16) * STRIDE + chunk % uint(BK / 16), a_chunks[i]);
            }
          }
          [unroll]
          for (int i = 0; i < B_RUNS; i++) {
            let run = tid + uint(i * THREADS);
            if (run < uint(BN * BK / 32)) {
              let word = base + uint(BM) * STRIDE + run / uint(BK / 32) * STRIDE + run % uint(BK / 32) * 2;
              S.write(word, b_chunks[2 * i]);
              S.write(word + 1, b_chunks[2 * i + 1]);
            }
          }
          if (ELEMENT_FOLD) {
            [unroll]
            for (int i = 0; i < M_CHUNKS; i++) {
              let entry = tid + uint(i * THREADS);
              let index = entry % uint(BM + BN);
              let slot = (stage & 1) * 2 + entry / uint(BM + BN);
              if (entry < uint(2 * (BM + BN))) {
                if (index < uint(BM)) {
                  int8_fold_rows[slot * 128 + index] = metadata_chunks[i].x;
                  int8_fold_rows[slot * 128 + 64 + index] = metadata_chunks[i].y;
                } else {
                  int8_fold_cols[slot * 128 + index - uint(BM)] = metadata_chunks[i].x;
                  int8_fold_cols[slot * 128 + 64 + index - uint(BM)] = metadata_chunks[i].y;
                }
              }
            }
          }
        }
        GroupMemoryBarrierWithGroupSync();
      }
    }
    if (ELEMENT_FOLD && split_phase != 2) {
      // Each accumulator once through the stages' memory, which no longer holds operands, into the lane's 8 outputs.
      [unroll]
      for (int i = 0; i < TM; i++) {
        [unroll]
        for (int j = 0; j < TN; j++) {
          GroupMemoryBarrierWithGroupSync();
          S.store_float(folded[i][j], subgroup * 64, 4);
          GroupMemoryBarrierWithGroupSync();
          let low = S.read(subgroup * 64 + lane * 2);
          let high = S.read(subgroup * 64 + lane * 2 + 1);
          [unroll]
          for (uint value = 0; value < 4; value++) {
            sums[i][j][value] = asfloat(low[value]);
            sums[i][j][4 + value] = asfloat(high[value]);
          }
        }
      }
    }

    let transform = has_rht && split_phase != 1;
    if (transform) {
      // Every subgroup has read its folded sums out of the stages' memory.
      GroupMemoryBarrierWithGroupSync();
    }
    [unroll]
    for (int i = 0; i < TM; i++) {
      let row_local = (subgroup / uint(SG_N)) * uint(TM * 16) + uint(16 * i) + lane / 2;
      let row = block_row + row_local;
      [unroll]
      for (int j = 0; j < TN; j++) {
        let col_local = (subgroup % uint(SG_N)) * uint(TN * 16) + uint(16 * j) + lane % 2 * 8;
        uint staged[8];
        [unroll]
        for (uint value = 0; value < 8; value++) {
          let col = block_col + col_local + value;
          staged[value] = 0;
          if (row < m && col < n && split_phase == 1) {
            partials[(split_index * m + row) * n + col] = asuint(sums[i][j][value]);
          } else if (row < m && col < n) {
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
        if (transform) {
          let word = row_local * uint(BN / 4) + col_local / 4;
          S.write(word, uint4(staged[0], staged[1], staged[2], staged[3]));
          S.write(word + 1, uint4(staged[4], staged[5], staged[6], staged[7]));
        }
      }
    }
    if (transform) {
      GroupMemoryBarrierWithGroupSync();
      // A subgroup per row and 32-column block in turn; n is whole blocks, so a block lies wholly in or past it.
      let blocks = uint(BN) / 32;
      for (uint cell = subgroup; cell < uint(BM) * blocks; cell += uint(SG_M * SG_N)) {
        let row_local = cell / blocks;
        let col_local = cell % blocks * 32 + lane;
        let row = block_row + row_local;
        let col = block_col + col_local;
        if (row < m && block_col + cell % blocks * 32 < n) {
          let value = asfloat(S.read(row_local * uint(BN / 4) + col_local / 4)[col_local % 4]);
          var output = hadamard_factor(subgroup_hadamard_transform(value), rht_factors[col]);
          if (rht_bias) {
            output = D.fromFloat(output).toFloat() + output_bias[col].toFloat();
          }
          d[row * n + col] = D.fromFloat(output);
        }
      }
    }
  }
}
