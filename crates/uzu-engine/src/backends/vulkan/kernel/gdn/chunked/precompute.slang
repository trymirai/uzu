import definitions;
import generated.delta_net;

static const uint CHUNKED_PRECOMPUTE_THREADS = 128;
static const uint CHUNKED_PRECOMPUTE_DEPTH = 16;
static const uint CHUNKED_PRECOMPUTE_STAGE_STRIDE = CHUNKED_PRECOMPUTE_DEPTH + 1;
static const uint CHUNKED_PRECOMPUTE_TILE_ROWS = 4;
static const uint CHUNKED_PRECOMPUTE_TILE_COLS = 8;
static const uint CHUNKED_PRECOMPUTE_TRIANGLE = DELTA_NET_CHUNKED_CHUNK * (DELTA_NET_CHUNKED_CHUNK - 1) / 2;

groupshared float chunked_precompute_shared[2 * CHUNKED_PRECOMPUTE_TRIANGLE];
groupshared float chunked_precompute_g[DELTA_NET_CHUNKED_MAX_GROUPS * DELTA_NET_CHUNKED_CHUNK];
groupshared float chunked_precompute_beta[DELTA_NET_CHUNKED_MAX_GROUPS * DELTA_NET_CHUNKED_CHUNK];

[[Kernel]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "(num_v_heads >= num_k_heads && num_v_heads <= crate::backends::vulkan::gpu_types::delta_net::DELTA_NET_CHUNKED_MAX_GROUPS * num_k_heads)")]]
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
  [[Groups("suffix_len.div_ceil(crate::backends::vulkan::gpu_types::delta_net::DELTA_NET_CHUNKED_CHUNK)")]] uint chunk,
  [[Groups("num_k_heads")]] uint hk,
  [[Threads("128")]] uint thread_idx,
) {
  let token_base = chunk * DELTA_NET_CHUNKED_CHUNK;
  let valid_tokens = min(DELTA_NET_CHUNKED_CHUNK, suffix_len - token_base);
  let groups_per_head = num_v_heads / num_k_heads;

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

    let group = pair + thread_idx / DELTA_NET_CHUNKED_CHUNK;
    let column = thread_idx % DELTA_NET_CHUNKED_CHUNK;
    if (group < groups_per_head) {
      let head = (chunk * num_v_heads + hk * groups_per_head + group) * DELTA_NET_CHUNKED_CHUNK * DELTA_NET_CHUNKED_CHUNK;
      let triangle = thread_idx / DELTA_NET_CHUNKED_CHUNK * CHUNKED_PRECOMPUTE_TRIANGLE;
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
