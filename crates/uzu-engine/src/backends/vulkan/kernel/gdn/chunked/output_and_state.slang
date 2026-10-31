import definitions;
import generated.delta_net;

// Per (v head, 16-value slice), with the state held as S^T[key, value], per 64-token chunk (padded tokens +0, unstored):
// R = beta (V - e^g (K S^T)); Vnew = T R over each token's causal 16-block; out = T(e^g (Q S^T) + qk_scaled Vnew);
// S^T = e^g_last S^T + (K e^(g_last - g))^T Vnew.
//
// The first 128 invocations work on the K side (K S^T, R, Vnew), the others on the Q side (Q S^T, out), each owning a
// 2-token x 4-value tile of its matrix; the state update has an invocation per key and half of the values. Keys and queries
// stage through shared memory CHUNKED_OUTPUT_DEPTH keys at a time, then T and qk_scaled, in float4s, rows padded against
// bank conflicts.

static const uint CHUNKED_OUTPUT_THREADS = 256;
// Values of a workgroup, as four float4 groups; tokens per invocation.
static const uint CHUNKED_OUTPUT_VT = 16;
static const uint CHUNKED_OUTPUT_TM = 2;
static const uint CHUNKED_OUTPUT_DEPTH = 32;
static const uint CHUNKED_OUTPUT_UPDATE_BATCH = 16;
// float4s per staged row of a key slice, and of T or qk_scaled.
static const uint CHUNKED_OUTPUT_SLICE_STRIDE = CHUNKED_OUTPUT_DEPTH / 4 + 1;
static const uint CHUNKED_OUTPUT_MATRIX_STRIDE = DELTA_NET_CHUNKED_CHUNK / 4 + 1;

// S^T[key][value group]; R, then Vnew, [token][value group]; the staged keys then queries [token][depth group], then
// T, then qk_scaled [token][column group]; each token's beta, e^g and e^(g_last - g).
groupshared float4 chunked_state_tile[DELTA_NET_HEAD_K_DIM * 4];
groupshared float4 chunked_new_value[DELTA_NET_CHUNKED_CHUNK * 4];
groupshared float4 chunked_stage[2 * DELTA_NET_CHUNKED_CHUNK * CHUNKED_OUTPUT_SLICE_STRIDE];
groupshared float chunked_row_beta[DELTA_NET_CHUNKED_CHUNK];
groupshared float chunked_row_exp_g[DELTA_NET_CHUNKED_CHUNK];
groupshared float chunked_row_decay[DELTA_NET_CHUNKED_CHUNK];

[[Variants("T", "float, bf16")]]
[[Kernel]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "num_v_heads == 0 || num_v_heads >= num_k_heads")]]
[[Precondition("encode", "head_v_dim.is_multiple_of(16)")]]
[[Precondition("encode", "suffix_len <= 1 || num_v_heads == 0 || head_v_dim == 0 || num_v_heads <= value_dim / head_v_dim")]]
func DeltaNetChunkedOutputAndState<T: IStorageFloat>(
  Ptr<float, Access.Read> q_norm,
  Ptr<float, Access.Read> k_norm,
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> qk_scaled,
  Ptr<float, Access.Read> t_mat,
  Ptr<float, Access.Read> g,
  Ptr<float, Access.Read> beta,
  Ptr<float, Access.ReadWrite> state,
  Ptr<T, Access.ReadWrite> out,
  uint num_v_heads,
  uint num_k_heads,
  uint head_v_dim,
  uint key_dim,
  uint value_dim,
  uint suffix_len,
  [[Groups("num_v_heads")]] uint hv_group,
  [[Groups("head_v_dim / 16")]] uint value_tile_group,
  [[Threads("256")]] uint thread_idx,
) {
  let hv = hv_group;
  let chunks = suffix_len / DELTA_NET_CHUNKED_CHUNK + (suffix_len % DELTA_NET_CHUNKED_CHUNK != 0 ? 1u : 0u);
  let conv_dim = 2 * key_dim + value_dim;
  let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  let k_head = hv / (num_v_heads / num_k_heads) * DELTA_NET_HEAD_K_DIM;
  let g_head = hv * suffix_len;
  let value_base = value_tile_group * CHUNKED_OUTPUT_VT;
  // The tile of this invocation's side (0: K, 1: Q): tokens row0 and row0 + 1, value group `value_group` of the slice.
  let side = thread_idx / 128;
  let row0 = (thread_idx % 128) / 4 * CHUNKED_OUTPUT_TM;
  let value_group = thread_idx % 4;
  // The state update's key and half (two value groups) of the slice.
  let key = thread_idx % DELTA_NET_HEAD_K_DIM;
  let half_groups = thread_idx / DELTA_NET_HEAD_K_DIM * 2;

  for (uint c = 0; c < 2; c++) {
    let value = (half_groups + c) * 4;
    float4 loaded;
    [unroll]
    for (uint j = 0; j < 4; j++) {
      loaded[j] = state[(hv * head_v_dim + value_base + value + j) * DELTA_NET_HEAD_K_DIM + key];
    }
    chunked_state_tile[key * 4 + half_groups + c] = loaded;
  }
  for (uint chunk = 0; chunk < chunks; chunk++) {
    let token_base = chunk * DELTA_NET_CHUNKED_CHUNK;
    let valid_tokens = min(DELTA_NET_CHUNKED_CHUNK, suffix_len - token_base);
    let chunk_head = (chunk * num_v_heads + hv) * DELTA_NET_CHUNKED_CHUNK * DELTA_NET_CHUNKED_CHUNK;
    let g_last = g[g_head + token_base + valid_tokens - 1];
    // The tokens a 16-token block's rows read from T and qk_scaled.
    let block_end = row0 / DELTA_NET_CHUNKED_BLOCK * DELTA_NET_CHUNKED_BLOCK + DELTA_NET_CHUNKED_BLOCK;

    if (thread_idx < DELTA_NET_CHUNKED_CHUNK) {
      var row_beta = 0.0;
      var row_exp_g = 0.0;
      var row_decay = 0.0;
      if (thread_idx < valid_tokens) {
        let token = token_base + thread_idx;
        let g_token = g[g_head + token];
        row_beta = beta[token * num_v_heads + hv];
        row_exp_g = exp(g_token);
        row_decay = exp(g_last - g_token);
      }
      chunked_row_beta[thread_idx] = row_beta;
      chunked_row_exp_g[thread_idx] = row_exp_g;
      chunked_row_decay[thread_idx] = row_decay;
    }

    // K S^T on the K side, Q S^T on the Q side, over the keys a slice at a time.
    float4 projection[CHUNKED_OUTPUT_TM] = { float4(0.0), float4(0.0) };
    let operand = side * DELTA_NET_CHUNKED_CHUNK * CHUNKED_OUTPUT_SLICE_STRIDE;
    for (uint depth_base = 0; depth_base < DELTA_NET_HEAD_K_DIM; depth_base += CHUNKED_OUTPUT_DEPTH) {
      GroupMemoryBarrierWithGroupSync();
      for (uint entry = thread_idx; entry < DELTA_NET_CHUNKED_CHUNK * (CHUNKED_OUTPUT_DEPTH / 4); entry += CHUNKED_OUTPUT_THREADS) {
        let t = entry / (CHUNKED_OUTPUT_DEPTH / 4);
        let q = entry % (CHUNKED_OUTPUT_DEPTH / 4);
        var key_values = float4(0.0);
        var query_values = float4(0.0);
        if (t < valid_tokens) {
          let source = (token_base + t) * key_dim + k_head + depth_base + q * 4;
          key_values = load_aligned<float4, 16>(k_norm, source);
          query_values = load_aligned<float4, 16>(q_norm, source);
        }
        chunked_stage[t * CHUNKED_OUTPUT_SLICE_STRIDE + q] = key_values;
        chunked_stage[DELTA_NET_CHUNKED_CHUNK * CHUNKED_OUTPUT_SLICE_STRIDE + t * CHUNKED_OUTPUT_SLICE_STRIDE + q] = query_values;
      }
      GroupMemoryBarrierWithGroupSync();
      for (uint q = 0; q < CHUNKED_OUTPUT_DEPTH / 4; q++) {
        let rows0 = chunked_stage[operand + row0 * CHUNKED_OUTPUT_SLICE_STRIDE + q];
        let rows1 = chunked_stage[operand + (row0 + 1) * CHUNKED_OUTPUT_SLICE_STRIDE + q];
        [unroll]
        for (uint k = 0; k < 4; k++) {
          let values = chunked_state_tile[(depth_base + q * 4 + k) * 4 + value_group];
          projection[0] = fma(float4(rows0[k]), values, projection[0]);
          projection[1] = fma(float4(rows1[k]), values, projection[1]);
        }
      }
    }

    // R = beta (V - e^g K S^T).
    if (side == 0) {
      [unroll]
      for (uint i = 0; i < CHUNKED_OUTPUT_TM; i++) {
        let t = row0 + i;
        var residual = float4(0.0);
        if (t < valid_tokens) {
          let source = (token_base + t) * total_proj_dim + 2 * key_dim + hv * head_v_dim + value_base + value_group * 4;
          let value_input = load_float4(in_proj, source);
          residual = chunked_row_beta[t] * (value_input - chunked_row_exp_g[t] * projection[i]);
        }
        chunked_new_value[t * 4 + value_group] = residual;
      }
    }
    GroupMemoryBarrierWithGroupSync();

    // Vnew = T R over each token's causal blocks; rows past the suffix are +0.
    for (uint entry = thread_idx; entry < DELTA_NET_CHUNKED_CHUNK * (DELTA_NET_CHUNKED_CHUNK / 4); entry += CHUNKED_OUTPUT_THREADS) {
      let t = entry / (DELTA_NET_CHUNKED_CHUNK / 4);
      let q = entry % (DELTA_NET_CHUNKED_CHUNK / 4);
      var causal = float4(0.0);
      if (t < valid_tokens) {
        causal = load_aligned<float4, 16>(t_mat, chunk_head + t * DELTA_NET_CHUNKED_CHUNK + q * 4);
      }
      chunked_stage[t * CHUNKED_OUTPUT_MATRIX_STRIDE + q] = causal;
    }
    GroupMemoryBarrierWithGroupSync();
    float4 new_value[CHUNKED_OUTPUT_TM] = { float4(0.0), float4(0.0) };
    if (side == 0) {
      for (uint q = 0; q < block_end / 4; q++) {
        let causal0 = chunked_stage[row0 * CHUNKED_OUTPUT_MATRIX_STRIDE + q];
        let causal1 = chunked_stage[(row0 + 1) * CHUNKED_OUTPUT_MATRIX_STRIDE + q];
        [unroll]
        for (uint k = 0; k < 4; k++) {
          let residual = chunked_new_value[(q * 4 + k) * 4 + value_group];
          new_value[0] = fma(float4(causal0[k]), residual, new_value[0]);
          new_value[1] = fma(float4(causal1[k]), residual, new_value[1]);
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if (side == 0) {
      chunked_new_value[row0 * 4 + value_group] = new_value[0];
      chunked_new_value[(row0 + 1) * 4 + value_group] = new_value[1];
    }
    for (uint entry = thread_idx; entry < DELTA_NET_CHUNKED_CHUNK * (DELTA_NET_CHUNKED_CHUNK / 4); entry += CHUNKED_OUTPUT_THREADS) {
      let t = entry / (DELTA_NET_CHUNKED_CHUNK / 4);
      let q = entry % (DELTA_NET_CHUNKED_CHUNK / 4);
      var scores = float4(0.0);
      if (t < valid_tokens) {
        scores = load_aligned<float4, 16>(qk_scaled, chunk_head + t * DELTA_NET_CHUNKED_CHUNK + q * 4);
      }
      chunked_stage[t * CHUNKED_OUTPUT_MATRIX_STRIDE + q] = scores;
    }
    GroupMemoryBarrierWithGroupSync();

    // out = e^g Q S^T + qk_scaled Vnew.
    if (side == 1) {
      float4 output[CHUNKED_OUTPUT_TM];
      [unroll]
      for (uint i = 0; i < CHUNKED_OUTPUT_TM; i++) {
        output[i] = projection[i] * chunked_row_exp_g[row0 + i];
      }
      for (uint q = 0; q < block_end / 4; q++) {
        let scores0 = chunked_stage[row0 * CHUNKED_OUTPUT_MATRIX_STRIDE + q];
        let scores1 = chunked_stage[(row0 + 1) * CHUNKED_OUTPUT_MATRIX_STRIDE + q];
        [unroll]
        for (uint k = 0; k < 4; k++) {
          let vnew = chunked_new_value[(q * 4 + k) * 4 + value_group];
          output[0] = fma(float4(scores0[k]), vnew, output[0]);
          output[1] = fma(float4(scores1[k]), vnew, output[1]);
        }
      }
      [unroll]
      for (uint i = 0; i < CHUNKED_OUTPUT_TM; i++) {
        let t = row0 + i;
        if (t < valid_tokens) {
          let destination = (token_base + t) * value_dim + hv * head_v_dim + value_base + value_group * 4;
          [unroll]
          for (uint j = 0; j < 4; j++) {
            out[destination + j] = T.fromFloat(output[i][j]);
          }
        }
      }
    }

    // S^T = e^g_last S^T + (K e^(g_last - g))^T Vnew. The keys load a batch at a time, so their latencies overlap; tokens
    // past the suffix scale to +0 against Vnew +0.
    float4 update[2] = { float4(0.0), float4(0.0) };
    for (uint batch = 0; batch < valid_tokens; batch += CHUNKED_OUTPUT_UPDATE_BATCH) {
      float scaled_keys[CHUNKED_OUTPUT_UPDATE_BATCH];
      [unroll]
      for (uint i = 0; i < CHUNKED_OUTPUT_UPDATE_BATCH; i++) {
        let s = batch + i;
        scaled_keys[i] = s < valid_tokens ? k_norm[(token_base + s) * key_dim + k_head + key] * chunked_row_decay[s] : 0.0;
      }
      [unroll]
      for (uint i = 0; i < CHUNKED_OUTPUT_UPDATE_BATCH; i++) {
        let s = batch + i;
        update[0] = fma(chunked_new_value[s * 4 + half_groups], float4(scaled_keys[i]), update[0]);
        update[1] = fma(chunked_new_value[s * 4 + half_groups + 1], float4(scaled_keys[i]), update[1]);
      }
    }
    let state_decay = exp(g_last);
    [unroll]
    for (uint c = 0; c < 2; c++) {
      let index = key * 4 + half_groups + c;
      chunked_state_tile[index] = state_decay * chunked_state_tile[index] + update[c];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  for (uint c = 0; c < 2; c++) {
    let value = (half_groups + c) * 4;
    let stored = chunked_state_tile[key * 4 + half_groups + c];
    [unroll]
    for (uint j = 0; j < 4; j++) {
      state[(hv * head_v_dim + value_base + value + j) * DELTA_NET_HEAD_K_DIM + key] = stored[j];
    }
  }
}
