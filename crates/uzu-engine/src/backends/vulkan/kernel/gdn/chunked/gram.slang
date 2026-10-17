import definitions;
import common.storage_address;
import matmul.common.fragment;
import matmul.common.simdgroup_fragment_ops;
import matmul.common.simdgroup_multiply_accumulate;

// Metal's DeltaNetChunkedGram, the q/k rows of token t and k head hk at t key_dim + hk 128. One 32-lane simdgroup per
// (chunk, k head, 16 x 32 tile) loads, per 8-deep slice of the keys, the tile's k and q rows and its k columns as FP32
// fragments, rows and columns past the suffix +0, and accumulates K K^T and Q K^T with fragment_mma, both sharing the
// k columns. A tile with valid rows and columns, whose first column lies before the end of its valid rows, is stored
// whole: kk, and for each grouped v head hv = hk floor(H / G) + group, qk_scaled[r, c] = (q_r . k_c) exp(g_r - g_c)
// where c <= r < valid tokens, otherwise +0. Other tiles stay untouched. Workgroups stride over chunks and heads by the
// actual NumWorkgroups, which the host clamps to the device limits. Element offsets are termwise 64-bit. The workgroup
// is exactly one logical simdgroup, as fragment_mma requires off native subgroups.

static const uint CHUNKED_GRAM_ROW_TILE = 16;
static const uint CHUNKED_GRAM_COL_TILE = 32;

[[Kernel]]
[[Precondition("new", "HEAD_K_DIM == 128")]]
[[Precondition("new", "CHUNK_SIZE == 32 || CHUNK_SIZE == 64")]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "num_v_heads == 0 || num_v_heads >= num_k_heads")]]
func DeltaNetChunkedGram(
  Ptr<float, Access.Read> q_norm,
  Ptr<float, Access.Read> k_norm,
  Ptr<float, Access.Read> g,
  Ptr<float, Access.ReadWrite> kk_out,
  Ptr<float, Access.ReadWrite> qk_scaled_out,
  uint num_v_heads,
  uint num_k_heads,
  uint key_dim,
  uint suffix_len,
  [[Groups("suffix_len.div_ceil(CHUNK_SIZE).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0])")]] uint chunk_group,
  [[Groups("num_k_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint hk_group,
  [[Groups("CHUNK_SIZE.div_ceil(16) * CHUNK_SIZE.div_ceil(32)")]] uint tile_idx,
  [[Threads("32")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
  [[Specialize]] uint CHUNK_SIZE,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let context = simdgroup_thread_context(thread_idx, 1);
  // This lane's first element (column, row) of each fragment.
  let lane = simdgroup_lane_coordinates(context.simd_lane_id);
  let col_tiles = CHUNK_SIZE / CHUNKED_GRAM_COL_TILE;
  let row_base = tile_idx / col_tiles * CHUNKED_GRAM_ROW_TILE;
  let col_base = tile_idx % col_tiles * CHUNKED_GRAM_COL_TILE;
  let tile_rows = min(CHUNKED_GRAM_ROW_TILE, CHUNK_SIZE - row_base);
  let tile_cols = min(CHUNKED_GRAM_COL_TILE, CHUNK_SIZE - col_base);
  let tile_dims = int2(int(tile_cols), int(tile_rows)) - lane;
  let lane_offset = (row_base + uint(lane.y)) * CHUNK_SIZE + col_base + uint(lane.x);
  let chunks = suffix_len / CHUNK_SIZE + (suffix_len % CHUNK_SIZE != 0 ? 1u : 0u);
  let groups_per_head = num_v_heads / num_k_heads;
  // Each stride advances only while more than one step remains, so it never wraps.
  for (var hk = hk_group; hk < num_k_heads; hk = num_k_heads - hk > groups.y ? hk + groups.y : num_k_heads) {
    for (var chunk = chunk_group; chunk < chunks; chunk = chunks - chunk > groups.x ? chunk + groups.x : chunks) {
      let chunk_token_base = chunk * CHUNK_SIZE;
      let row_token_base = chunk_token_base + row_base;
      let col_token_base = chunk_token_base + col_base;
      let valid_rows = row_token_base < suffix_len ? min(tile_rows, suffix_len - row_token_base) : 0u;
      let valid_cols = col_token_base < suffix_len ? min(tile_cols, suffix_len - col_token_base) : 0u;
      // Upper-causal tiles are zero.
      if (valid_rows == 0 || valid_cols == 0 || col_base >= row_base + valid_rows) {
        continue;
      }

      Fragment<float, 2, 4> kk_acc;
      Fragment<float, 2, 4> qk_acc;
      kk_acc.clear();
      qk_acc.clear();
      let row_element = element_index(row_token_base + uint(lane.y), key_dim, uint(lane.x));
      let col_element = element_index(col_token_base + uint(lane.x), key_dim, uint(lane.y));
      let left_bounds = int2(8, int(valid_rows)) - lane;
      // The k columns read transposed: rows along the keys, columns along the tokens.
      let right_bounds = int2(int(valid_cols), 8) - lane;
      for (uint k_block_start = 0; k_block_start < HEAD_K_DIM; k_block_start += 8) {
        let k_head = element_index(hk, HEAD_K_DIM, k_block_start);
        let k_rows = element_pointer(element_pointer(k_norm, row_element), k_head);
        let q_rows = element_pointer(element_pointer(q_norm, row_element), k_head);
        let k_cols = element_pointer(element_pointer(k_norm, col_element), k_head);
        // gram.h's accumulate_dual_gram_tile.
        Fragment<float, 2, 1> k_left;
        Fragment<float, 2, 1> q_left;
        Fragment<float, 1, 4> k_right;
        if (valid_rows == tile_rows) {
          k_left.load_from<false>(k_rows, key_dim, 1, left_bounds);
          q_left.load_from<false>(q_rows, key_dim, 1, left_bounds);
        } else {
          k_left.load_from<true>(k_rows, key_dim, 1, left_bounds);
          q_left.load_from<true>(q_rows, key_dim, 1, left_bounds);
        }
        if (valid_cols == tile_cols) {
          k_right.load_from<false>(k_cols, 1, key_dim, right_bounds);
        } else {
          k_right.load_from<true>(k_cols, 1, key_dim, right_bounds);
        }
        fragment_mma(kk_acc, k_left, k_right, context);
        fragment_mma(qk_acc, q_left, k_right, context);
      }

      let kk_tile = element_pointer(kk_out, byte_offset(element_index(chunk, num_k_heads, hk), CHUNK_SIZE * CHUNK_SIZE));
      fragment_store_safe<float, 1, 1, 2, 4>(kk_acc, element_pointer(kk_tile, uint2(lane_offset, 0)), CHUNK_SIZE, tile_dims);

      // Store qk for each grouped v head with decay scale and causal mask.
      let valid_tokens = min(CHUNK_SIZE, suffix_len - chunk_token_base);
      for (uint group = 0; group < groups_per_head; ++group) {
        let hv = hk * groups_per_head + group;
        let g_head = element_pointer(g, element_index(hv, suffix_len, chunk_token_base));
        var scaled = qk_acc;
        [unroll]
        for (int m = 0; m < 2; ++m) {
          [unroll]
          for (int n = 0; n < 4; ++n) {
            [unroll]
            for (int element = 0; element < 2; ++element) {
              let row = row_base + uint(lane.y + 8 * m);
              let col = col_base + uint(lane.x + 8 * n + element);
              var value = 0.0;
              if (row < valid_tokens && col < valid_tokens && col <= row) {
                value = scaled.fragment_data[m * 4 + n][element] * exp(g_head[row] - g_head[col]);
              }
              scaled.fragment_data[m * 4 + n][element] = value;
            }
          }
        }
        let qk_tile = element_pointer(qk_scaled_out, byte_offset(element_index(chunk, num_v_heads, hv), CHUNK_SIZE * CHUNK_SIZE));
        fragment_store_safe<float, 1, 1, 2, 4>(scaled, element_pointer(qk_tile, uint2(lane_offset, 0)), CHUNK_SIZE, tile_dims);
      }
    }
  }
}
