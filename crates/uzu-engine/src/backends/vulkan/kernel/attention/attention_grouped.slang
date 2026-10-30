import definitions;
import attention.mask;
import generated.attention_grouped;
import generated.ring;
import generated.trie;

using namespace linalg;

// AttentionTwoPass1 on cooperative matrices for the queries of a key-value head together: its gqa_factor query heads'
// suffix_length queries are rows head_in_group * suffix_length + query, contiguous in `queries` and in the partials. A
// workgroup takes a 16-row tile over split `split` of the 64-key blocks; per block, subgroup s scores key tile s % 4
// over half s / 4 of the head dimensions, owns rows 2s and 2s + 1 for the softmax and head dimensions
// [s HEAD_DIM / 8, (s + 1) HEAD_DIM / 8) for the values. A split of one block scores it once; longer ones score their
// blocks twice, first for each row's largest admitted score, then for the exponentials, their sum and their product
// with the values. The last row tile and the last key block end at the data: the tile repeats rows an earlier tile
// took, rewriting their identical partials, and the block skips keys.

static const uint GROUPED_SUBGROUPS = 8;
static const uint GROUPED_KEY_TILES = ATTENTION_GROUPED_KEYS / 16;
static const uint GROUPED_SUBGROUP_ROWS = ATTENTION_GROUPED_ROWS / GROUPED_SUBGROUPS;

typealias GroupedTile = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

// The block's scores [depth half][16 rows][64 keys] and their exponentials as stored.
groupshared float grouped_scores[2 * ATTENTION_GROUPED_ROWS * ATTENTION_GROUPED_KEYS];
groupshared uint16_t grouped_weights[ATTENTION_GROUPED_ROWS * ATTENTION_GROUPED_KEYS];

struct GroupedArguments<T : IModelFloat16> {
  Ptr<T, Access.Read> queries;
  Ptr<T, Access.Read> keys;
  Ptr<T, Access.Read> values;
  Ptr<float, Access.ReadWrite> partials;
  Ptr<float, Access.ReadWrite> sums;
  Ptr<float, Access.ReadWrite> maxs;
  uint gqa_factor;
  uint sequence_length;
  uint k_head_stride;
  uint k_seq_stride;
  uint v_head_stride;
  uint v_seq_stride;
  RingParams ring_params;
  float scale;
  uint suffix_length;
  Ptr<TrieNode, Access.Read> trie;
  uint sliding_window_size;
  uint splits;
  bool is_kv_cache_ring;
  bool is_causal;
  bool is_trie;
  bool is_sliding_window;

  // Whether grouped row `row` admits `key`, by attention_use_key of its query.
  bool admits(uint row, uint key) {
    let query = row % suffix_length;
    let prefix_length = sequence_length - suffix_length;
    let suffix_position = is_kv_cache_ring ? ring_params.ring_length : prefix_length;
    var query_position = suffix_position + query;
    if (is_trie && is_sliding_window) {
      query_position = suffix_position + trie[query].height;
    }
    return attention_use_key(
      key, query, prefix_length, suffix_position, query_position, ring_params, trie, sliding_window_size,
      is_kv_cache_ring, is_causal, is_trie, is_sliding_window
    );
  }

  // Element `element` of a 16-bit operand as a 16 x 16 matrix, rows `stride` apart. Slang's CoopMat.Load takes
  // read-write device pointers only: the access is Slang's, the SPIR-V pointer type the same, so the address is copied
  // rather than converted, which would widen it through 64 bits; only BF16 operands then change element type.
  static CoopMat<N, MemoryScope.Subgroup, 16, 16, USE> load<
    N : ICoopElement,
    let USE : CoopMatMatrixUse,
    let LAYOUT : CoopMatMatrixLayout
  >(Ptr<T, Access.Read> operand, uint element, uint stride) {
    let writable = spirv_asm { result:$$Ptr<T, Access.ReadWrite> = OpCopyObject $operand };
    let pointer = bit_cast<Ptr<N, Access.ReadWrite>>(writable);
    return CoopMat<N, MemoryScope.Subgroup, 16, 16, USE>.Load<LAYOUT>(pointer, element, stride);
  }
}

struct GroupedAttention<T : IModelFloat16, N : ICoopElement, let HEAD_DIM : int> {
  static const int DEPTH_TILES = HEAD_DIM / 32;
  static const int SLICE_TILES = HEAD_DIM / 16 / int(GROUPED_SUBGROUPS);

  GroupedArguments<T> arguments;
  // This subgroup's depth half of the queries.
  CoopMat<N, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA> queries[DEPTH_TILES];
  uint first_row;
  uint key_element;
  uint value_element;
  uint subgroup;

  // This subgroup's key tile and depth half of the block from `first_key`, into the scores.
  void score(uint first_key) {
    var sum = GroupedTile(0.0);
    let tile = subgroup % GROUPED_KEY_TILES;
    let depth = subgroup / GROUPED_KEY_TILES * uint(16 * DEPTH_TILES);
    let key_row = key_element + (first_key + 16 * tile) * arguments.k_seq_stride + depth;
    [unroll]
    for (int d = 0; d < DEPTH_TILES; d++) {
      // The keys are the columns of K^T.
      let k = GroupedArguments<T>.load<N, CoopMatMatrixUse.MatrixB, CoopMatMatrixLayout.ColumnMajor>(
        arguments.keys, key_row + uint(16 * d), arguments.k_seq_stride
      );
      sum = coopMatMulAdd<float, false>(queries[d], k, sum);
    }
    let half_scores = subgroup / GROUPED_KEY_TILES * ATTENTION_GROUPED_ROWS * ATTENTION_GROUPED_KEYS;
    sum.Store<CoopMatMatrixLayout.RowMajor, float>(grouped_scores, half_scores + 16 * tile, ATTENTION_GROUPED_KEYS);
  }

  // Lane l's scaled scores of keys l and l + 32 of the block from `first_key` in this subgroup's row `r`; keys the row
  // rejects and keys below `fresh` are -inf.
  float2 lane_scores(uint r, uint first_key, uint fresh) {
    let row = GROUPED_SUBGROUP_ROWS * subgroup + r;
    var scores = float2(NEGATIVE_INFINITY);
    [unroll]
    for (uint half = 0; half < 2; half++) {
      let column = WaveGetLaneIndex() + 32 * half;
      let key = first_key + column;
      if (key >= fresh && arguments.admits(first_row + row, key)) {
        // The two depth halves' partial scores.
        let element = row * ATTENTION_GROUPED_KEYS + column;
        let other_half = ATTENTION_GROUPED_ROWS * ATTENTION_GROUPED_KEYS + element;
        scores[half] = arguments.scale * (grouped_scores[element] + grouped_scores[other_half]);
      }
    }
    return scores;
  }

  static void run(GroupedArguments<T> arguments, uint row_tile, uint kv_head, uint split, uint subgroup) {
    let rows = arguments.gqa_factor * arguments.suffix_length;
    let sequence_length = arguments.sequence_length;
    let blocks = (sequence_length + ATTENTION_GROUPED_KEYS - 1) / ATTENTION_GROUPED_KEYS;
    let first_block = split * blocks / arguments.splits;
    let end_block = (split + 1) * blocks / arguments.splits;
    let lane = WaveGetLaneIndex();

    GroupedAttention<T, N, HEAD_DIM> attention;
    attention.arguments = arguments;
    attention.first_row = min(row_tile * ATTENTION_GROUPED_ROWS, rows - ATTENTION_GROUPED_ROWS);
    attention.key_element = kv_head * arguments.k_head_stride;
    attention.value_element = kv_head * arguments.v_head_stride + uint(16 * SLICE_TILES) * subgroup;
    attention.subgroup = subgroup;
    let query_element = (kv_head * rows + attention.first_row) * uint(HEAD_DIM)
      + subgroup / GROUPED_KEY_TILES * uint(16 * DEPTH_TILES);
    [unroll]
    for (int d = 0; d < DEPTH_TILES; d++) {
      attention.queries[d] = GroupedArguments<T>.load<N, CoopMatMatrixUse.MatrixA, CoopMatMatrixLayout.RowMajor>(
        arguments.queries, query_element + uint(16 * d), uint(HEAD_DIM)
      );
    }

    float row_max[GROUPED_SUBGROUP_ROWS];
    [unroll]
    for (uint r = 0; r < GROUPED_SUBGROUP_ROWS; r++) {
      row_max[r] = NEGATIVE_INFINITY;
    }
    for (uint block = first_block; block < end_block; block++) {
      let first_key = min(block * ATTENTION_GROUPED_KEYS, sequence_length - ATTENTION_GROUPED_KEYS);
      attention.score(first_key);
      GroupMemoryBarrierWithGroupSync();
      [unroll]
      for (uint r = 0; r < GROUPED_SUBGROUP_ROWS; r++) {
        let scores = attention.lane_scores(r, first_key, block * ATTENTION_GROUPED_KEYS);
        row_max[r] = max(row_max[r], WaveActiveMax(max(scores.x, scores.y)));
      }
      GroupMemoryBarrierWithGroupSync();
    }

    float row_sum[GROUPED_SUBGROUP_ROWS];
    GroupedTile outputs[SLICE_TILES];
    [unroll]
    for (uint r = 0; r < GROUPED_SUBGROUP_ROWS; r++) {
      row_sum[r] = 0.0;
    }
    [unroll]
    for (int d = 0; d < SLICE_TILES; d++) {
      outputs[d] = GroupedTile(0.0);
    }
    // A one-block split keeps its scores from the first pass.
    let rescore = end_block - first_block > 1;
    for (uint block = first_block; block < end_block; block++) {
      let first_key = min(block * ATTENTION_GROUPED_KEYS, sequence_length - ATTENTION_GROUPED_KEYS);
      if (rescore) {
        attention.score(first_key);
        GroupMemoryBarrierWithGroupSync();
      }
      [unroll]
      for (uint r = 0; r < GROUPED_SUBGROUP_ROWS; r++) {
        let scores = attention.lane_scores(r, first_key, block * ATTENTION_GROUPED_KEYS);
        var weights = float2(0.0);
        [unroll]
        for (uint half = 0; half < 2; half++) {
          if (scores[half] != NEGATIVE_INFINITY) {
            weights[half] = exp(scores[half] - row_max[r]);
          }
          let index = (GROUPED_SUBGROUP_ROWS * subgroup + r) * ATTENTION_GROUPED_KEYS + lane + 32 * half;
          grouped_weights[index] = uint16_t(T.fromFloat(weights[half]).storageBits());
        }
        row_sum[r] += WaveActiveSum(weights.x + weights.y);
      }
      GroupMemoryBarrierWithGroupSync();
      [unroll]
      for (int tile = 0; tile < int(GROUPED_KEY_TILES); tile++) {
        let p = CoopMat<N, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>.Load<
          CoopMatMatrixLayout.RowMajor, uint16_t
        >(grouped_weights, uint(16 * tile), ATTENTION_GROUPED_KEYS);
        let value_row = attention.value_element + (first_key + uint(16 * tile)) * arguments.v_seq_stride;
        [unroll]
        for (int d = 0; d < SLICE_TILES; d++) {
          let v = GroupedArguments<T>.load<N, CoopMatMatrixUse.MatrixB, CoopMatMatrixLayout.RowMajor>(
            arguments.values, value_row + uint(16 * d), arguments.v_seq_stride
          );
          outputs[d] = coopMatMulAdd<float, false>(p, v, outputs[d]);
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }

    // Partial row kv_head * rows + row is head * suffix_length + query.
    let partial_row = kv_head * rows + attention.first_row;
    [unroll]
    for (uint r = 0; r < GROUPED_SUBGROUP_ROWS; r++) {
      let partial = (partial_row + GROUPED_SUBGROUP_ROWS * subgroup + r) * arguments.splits + split;
      if (lane == 0) {
        arguments.sums[partial] = row_sum[r];
        arguments.maxs[partial] = row_max[r];
      }
    }
    let partial_element = (partial_row * arguments.splits + split) * uint(HEAD_DIM) + uint(16 * SLICE_TILES) * subgroup;
    [unroll]
    for (int d = 0; d < SLICE_TILES; d++) {
      outputs[d].Store<CoopMatMatrixLayout.RowMajor>(
        arguments.partials, partial_element + uint(16 * d), arguments.splits * uint(HEAD_DIM)
      );
    }
  }
}

[[Kernel]]
[[RequiredSubgroupSize("32")]]
[[Precondition("new", "context.physical_device().capabilities.float_tensor_cores(T)")]]
[[Precondition("new", "HEAD_DIM == 128 || HEAD_DIM == 256")]]
[[Precondition("encode", "gqa_factor > 0 && num_heads.is_multiple_of(gqa_factor) && gqa_factor * suffix_length >= 16")]]
[[Precondition("encode", "sequence_length >= 64 && suffix_length <= sequence_length")]]
[[Precondition("encode", "splits > 0 && splits <= sequence_length.div_ceil(64)")]]
func AttentionGrouped<T: IModelFloat16>(
  Ptr<T, Access.Read> queries,
  Ptr<T, Access.Read> keys,
  Ptr<T, Access.Read> values,
  Ptr<float, Access.ReadWrite> partials,
  Ptr<float, Access.ReadWrite> sums,
  Ptr<float, Access.ReadWrite> maxs,
  uint gqa_factor,
  uint sequence_length,
  uint k_head_stride,
  uint k_seq_stride,
  uint v_head_stride,
  uint v_seq_stride,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  float scale,
  uint num_heads,
  uint suffix_length,
  [[Optional("is_trie")]] Ptr<TrieNode, Access.Read> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  uint splits,
  [[Groups("(gqa_factor * suffix_length).div_ceil(16)")]] uint row_tile,
  [[Groups("num_heads / gqa_factor")]] uint kv_head,
  [[Groups("splits")]] uint split,
  [[Threads("32")]] uint lane,
  [[Threads("8")]] uint tid_y,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  GroupedArguments<T> arguments = {
    queries, keys, values, partials, sums, maxs, gqa_factor, sequence_length, k_head_stride, k_seq_stride,
    v_head_stride, v_seq_stride, ring_params, scale, suffix_length, trie, sliding_window_size, splits,
    is_kv_cache_ring, is_causal, is_trie, is_sliding_window
  };
  if (HEAD_DIM == 128) {
    GroupedAttention<T, BFloat16, 128>.run(arguments, row_tile, kv_head, split, subgroup_id());
  } else {
    GroupedAttention<T, BFloat16, 256>.run(arguments, row_tile, kv_head, split, subgroup_id());
  }
}
