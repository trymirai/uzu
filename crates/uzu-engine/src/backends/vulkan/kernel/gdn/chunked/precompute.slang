import definitions;
import generated.delta_net;

// One workgroup per (64-token chunk, k head), over the v heads that share that head's keys, one hv = hk G + group each:
//   g[hv, t]  the inclusive prefix of log_decay within the chunk;
//   qk_scaled  (q_r . k_c) e^(g_r - g_c) for c <= r;
//   T          (I + A)^-1 for A[r, c] = beta_r e^(g_r - g_c) (k_r . k_c), c < r.
// qk_scaled and T are [chunk, hv, 64, 64], written where DeltaNetChunkedOutputAndState reads them: the columns up to the
// end of each row's 16-token block, +0 above the diagonal. Rows and columns past the suffix count as +0.

static const uint CHUNKED_PRECOMPUTE_THREADS = 128;
// Keys and queries are staged for the Gram matrices this deep, rows padded by one.
static const uint CHUNKED_PRECOMPUTE_DEPTH = 16;
static const uint CHUNKED_PRECOMPUTE_STAGE_STRIDE = CHUNKED_PRECOMPUTE_DEPTH + 1;
// Each invocation owns a 4 x 8 tile of the Gram matrices; 8 divides DELTA_NET_CHUNKED_BLOCK.
static const uint CHUNKED_PRECOMPUTE_TILE_ROWS = 4;
static const uint CHUNKED_PRECOMPUTE_TILE_COLS = 8;
// A's strictly lower triangle, row r from r (r - 1) / 2.
static const uint CHUNKED_PRECOMPUTE_TRIANGLE = DELTA_NET_CHUNKED_CHUNK * (DELTA_NET_CHUNKED_CHUNK - 1) / 2;

// The staged keys, then queries, [token, depth] during the Gram matrices; then the triangles of A for the pair of v heads
// being solved, one per half of the workgroup.
groupshared float chunked_precompute_shared[2 * CHUNKED_PRECOMPUTE_TRIANGLE];
groupshared float chunked_precompute_g[DELTA_NET_CHUNKED_MAX_GROUPS * DELTA_NET_CHUNKED_CHUNK];
groupshared float chunked_precompute_beta[DELTA_NET_CHUNKED_MAX_GROUPS * DELTA_NET_CHUNKED_CHUNK];

[[Kernel]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "num_v_heads == 0 || (num_v_heads >= num_k_heads && num_v_heads <= 4 * num_k_heads)")]]
func DeltaNetChunkedPrecompute(
  Ptr<float, Access.Read> q_norm,
  Ptr<float, Access.Read> k_norm,
  Ptr<float, Access.Read> log_decay,
  Ptr<float, Access.Read> beta,
  Ptr<float, Access.ReadWrite> g_out,
  Ptr<float, Access.ReadWrite> qk_scaled_out,
  Ptr<float, Access.ReadWrite> t_out,
  uint num_v_heads,
  uint num_k_heads,
  uint key_dim,
  uint suffix_len,
  [[Groups("suffix_len.div_ceil(crate::backends::vulkan::gpu_types::delta_net::DELTA_NET_CHUNKED_CHUNK)")]] uint chunk_group,
  [[Groups("num_k_heads")]] uint hk_group,
  [[Threads("128")]] uint thread_idx,
) {
  let chunk = chunk_group;
  let hk = hk_group;
  let token_base = chunk * DELTA_NET_CHUNKED_CHUNK;
  let valid_tokens = min(DELTA_NET_CHUNKED_CHUNK, suffix_len - token_base);
  let groups_per_head = num_v_heads / num_k_heads;

  // One invocation per v head scans its chunk's decay.
  if (thread_idx < groups_per_head) {
    let hv = hk * groups_per_head + thread_idx;
    var prefix = 0.0;
    for (uint t = 0; t < DELTA_NET_CHUNKED_CHUNK; t++) {
      var decay = 0.0;
      var beta_t = 0.0;
      if (t < valid_tokens) {
        let offset = (token_base + t) * num_v_heads + hv;
        decay = log_decay[offset];
        beta_t = beta[offset];
      }
      prefix += decay;
      chunked_precompute_g[thread_idx * DELTA_NET_CHUNKED_CHUNK + t] = prefix;
      chunked_precompute_beta[thread_idx * DELTA_NET_CHUNKED_CHUNK + t] = beta_t;
      if (t < valid_tokens) {
        g_out[hv * suffix_len + token_base + t] = prefix;
      }
    }
  }

  // K K^T and Q K^T over the head dim, a 4 x 8 tile each; tiles wholly above the diagonal or past the suffix stay +0.
  let tile_cols_per_row = DELTA_NET_CHUNKED_CHUNK / CHUNKED_PRECOMPUTE_TILE_COLS;
  let row_base = thread_idx / tile_cols_per_row * CHUNKED_PRECOMPUTE_TILE_ROWS;
  let col_base = thread_idx % tile_cols_per_row * CHUNKED_PRECOMPUTE_TILE_COLS;
  let has_gram = row_base < valid_tokens && col_base < valid_tokens && col_base < row_base + CHUNKED_PRECOMPUTE_TILE_ROWS;
  float kk[CHUNKED_PRECOMPUTE_TILE_ROWS][CHUNKED_PRECOMPUTE_TILE_COLS];
  float qk[CHUNKED_PRECOMPUTE_TILE_ROWS][CHUNKED_PRECOMPUTE_TILE_COLS];
  [unroll]
  for (uint i = 0; i < CHUNKED_PRECOMPUTE_TILE_ROWS; i++) {
    [unroll]
    for (uint j = 0; j < CHUNKED_PRECOMPUTE_TILE_COLS; j++) {
      kk[i][j] = 0.0;
      qk[i][j] = 0.0;
    }
  }
  let query_stage = DELTA_NET_CHUNKED_CHUNK * CHUNKED_PRECOMPUTE_STAGE_STRIDE;
  let key_head = hk * DELTA_NET_HEAD_K_DIM;
  for (uint depth = 0; depth < DELTA_NET_HEAD_K_DIM; depth += CHUNKED_PRECOMPUTE_DEPTH) {
    GroupMemoryBarrierWithGroupSync();
    for (uint entry = thread_idx; entry < DELTA_NET_CHUNKED_CHUNK * CHUNKED_PRECOMPUTE_DEPTH; entry += CHUNKED_PRECOMPUTE_THREADS) {
      let t = entry / CHUNKED_PRECOMPUTE_DEPTH;
      let d = entry % CHUNKED_PRECOMPUTE_DEPTH;
      var key = 0.0;
      var query = 0.0;
      if (t < valid_tokens) {
        let source = (token_base + t) * key_dim + key_head + depth + d;
        key = k_norm[source];
        query = q_norm[source];
      }
      chunked_precompute_shared[t * CHUNKED_PRECOMPUTE_STAGE_STRIDE + d] = key;
      chunked_precompute_shared[query_stage + t * CHUNKED_PRECOMPUTE_STAGE_STRIDE + d] = query;
    }
    GroupMemoryBarrierWithGroupSync();
    if (has_gram) {
      for (uint d = 0; d < CHUNKED_PRECOMPUTE_DEPTH; d++) {
        float key_rows[CHUNKED_PRECOMPUTE_TILE_ROWS];
        float query_rows[CHUNKED_PRECOMPUTE_TILE_ROWS];
        float key_cols[CHUNKED_PRECOMPUTE_TILE_COLS];
        [unroll]
        for (uint i = 0; i < CHUNKED_PRECOMPUTE_TILE_ROWS; i++) {
          let stage = (row_base + i) * CHUNKED_PRECOMPUTE_STAGE_STRIDE + d;
          key_rows[i] = chunked_precompute_shared[stage];
          query_rows[i] = chunked_precompute_shared[query_stage + stage];
        }
        [unroll]
        for (uint j = 0; j < CHUNKED_PRECOMPUTE_TILE_COLS; j++) {
          key_cols[j] = chunked_precompute_shared[(col_base + j) * CHUNKED_PRECOMPUTE_STAGE_STRIDE + d];
        }
        [unroll]
        for (uint i = 0; i < CHUNKED_PRECOMPUTE_TILE_ROWS; i++) {
          [unroll]
          for (uint j = 0; j < CHUNKED_PRECOMPUTE_TILE_COLS; j++) {
            kk[i][j] = fma(key_rows[i], key_cols[j], kk[i][j]);
            qk[i][j] = fma(query_rows[i], key_cols[j], qk[i][j]);
          }
        }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // The v heads two at a time, each half of the workgroup solving one.
  let block_limit = (row_base / DELTA_NET_CHUNKED_BLOCK + 1) * DELTA_NET_CHUNKED_BLOCK;
  for (uint pair = 0; pair < groups_per_head; pair += 2) {
    for (uint slot = 0; slot < 2; slot++) {
      let group = pair + slot;
      if (group >= groups_per_head) {
        continue;
      }
      let head = (chunk * num_v_heads + hk * groups_per_head + group) * DELTA_NET_CHUNKED_CHUNK * DELTA_NET_CHUNKED_CHUNK;
      let triangle = slot * CHUNKED_PRECOMPUTE_TRIANGLE;
      for (uint i = 0; i < CHUNKED_PRECOMPUTE_TILE_ROWS; i++) {
        let row = row_base + i;
        for (uint j = 0; j < CHUNKED_PRECOMPUTE_TILE_COLS; j++) {
          let col = col_base + j;
          var scaled = 0.0;
          var a = 0.0;
          if (row < valid_tokens && col <= row) {
            let decay = exp(
              chunked_precompute_g[group * DELTA_NET_CHUNKED_CHUNK + row]
              - chunked_precompute_g[group * DELTA_NET_CHUNKED_CHUNK + col]
            );
            scaled = qk[i][j] * decay;
            a = chunked_precompute_beta[group * DELTA_NET_CHUNKED_CHUNK + row] * decay * kk[i][j];
          }
          if (col < row) {
            chunked_precompute_shared[triangle + row * (row - 1) / 2 + col] = a;
          }
          if (row < valid_tokens && col_base < block_limit) {
            qk_scaled_out[head + row * DELTA_NET_CHUNKED_CHUNK + col] = scaled;
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();

    // T[:, column] by forward substitution of (I + A) x = e_column, x = +0 above column.
    let group = pair + thread_idx / DELTA_NET_CHUNKED_CHUNK;
    let column = thread_idx % DELTA_NET_CHUNKED_CHUNK;
    if (group < groups_per_head) {
      let head = (chunk * num_v_heads + hk * groups_per_head + group) * DELTA_NET_CHUNKED_CHUNK * DELTA_NET_CHUNKED_CHUNK;
      let triangle = thread_idx / DELTA_NET_CHUNKED_CHUNK * CHUNKED_PRECOMPUTE_TRIANGLE;
      // A block's rows read the earlier rows of the column back from t_out, which this invocation stored: T is lower
      // triangular, so each is either +0 above the column or stored. Rows past the suffix have no A and stay unstored.
      let first_block = column / DELTA_NET_CHUNKED_BLOCK;
      let blocks = (valid_tokens + DELTA_NET_CHUNKED_BLOCK - 1) / DELTA_NET_CHUNKED_BLOCK;
      for (uint block = first_block; block < blocks; block++) {
        let base = block * DELTA_NET_CHUNKED_BLOCK;
        float y[DELTA_NET_CHUNKED_BLOCK];
        [unroll]
        for (uint i = 0; i < DELTA_NET_CHUNKED_BLOCK; i++) {
          y[i] = base + i == column ? 1.0 : 0.0;
        }
        for (uint k = first_block * DELTA_NET_CHUNKED_BLOCK; k < base; k++) {
          let earlier = t_out[head + k * DELTA_NET_CHUNKED_CHUNK + column];
          [unroll]
          for (uint i = 0; i < DELTA_NET_CHUNKED_BLOCK; i++) {
            let row = base + i;
            y[i] = fma(-chunked_precompute_shared[triangle + row * (row - 1) / 2 + k], earlier, y[i]);
          }
        }
        [unroll]
        for (uint i = 0; i < DELTA_NET_CHUNKED_BLOCK; i++) {
          let row = base + i;
          [unroll]
          for (uint j = 0; j < i; j++) {
            y[i] = fma(-chunked_precompute_shared[triangle + row * (row - 1) / 2 + base + j], y[j], y[i]);
          }
          if (row < valid_tokens) {
            t_out[head + row * DELTA_NET_CHUNKED_CHUNK + column] = y[i];
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
}
