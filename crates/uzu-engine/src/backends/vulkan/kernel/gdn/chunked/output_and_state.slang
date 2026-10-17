import definitions;
import common.activation;
import common.storage_address;
import gdn.prefill_prep;

// The Metal DeltaNetChunkedOutputAndState over 64-token chunks, one workgroup per (v head hv, 32-value slice), with
// hk = hv / floor(H / G), q and k of token t at t key_dim + hk 128, and v at in_proj[t total_proj_dim + 2 key_dim +
// hv head_v_dim + value]. The slice's state is held as S^T[key, value]; per chunk, with t < valid tokens:
// 1. R[t, v] = beta_t (V[t, v] - e^g_t (k_t . S^T[:, v])), +0 for padded t.
// 2. Vnew[t, v] = sum over s < 16 (t / 16 + 1) of T[t, s] R[s, v], T taken as +0 on padded rows.
// 3. out[t, v] = O((e^g_t (q_t . S^T[:, v])) + sum over the same s of qk_scaled[t, s] Vnew[s, v]), qk_scaled +0 for
//    padded s; padded rows are not stored.
// 4. S^T[j, v] = e^g_last S^T[j, v] + sum over all 64 s of Vnew[s, v] (k_s[j] e^(g_last - g_s)), both +0 for padded s.
// Dots and sums run from +0 (step 3 from its scaled query term) in ascending order; each product and sum is one IEEE
// FP32 operation, never fused. The state is read before the first chunk and written after the last; the update follows
// each chunk's output. Strides use the clamped NumWorkgroups, so every invocation reaches each barrier; element offsets
// are termwise 64-bit and total_proj_dim is never formed.

static const uint CHUNKED_OUTPUT_THREADS = 128;
static const uint CHUNKED_OUTPUT_CHUNK = 64;
static const uint CHUNKED_OUTPUT_TOKEN_TILE = 16;
static const uint CHUNKED_OUTPUT_VT = 32;
static const uint CHUNKED_OUTPUT_ENTRIES = CHUNKED_OUTPUT_CHUNK * CHUNKED_OUTPUT_VT / CHUNKED_OUTPUT_THREADS;

// S^T[key VT + value]; invocation j owns key j. R, then Vnew, [token VT + value], remain FP32.
groupshared float chunked_state_tile[128 * CHUNKED_OUTPUT_VT];
groupshared float chunked_scratch_tile[CHUNKED_OUTPUT_CHUNK * CHUNKED_OUTPUT_VT];

[[Kernel]]
[[Precondition("new", "HEAD_K_DIM == 128")]]
[[Precondition("new", "VT == 32")]]
[[Precondition("new", "!USE_MXU")]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "num_v_heads == 0 || num_v_heads >= num_k_heads")]]
[[Precondition("encode", "head_v_dim.is_multiple_of(32)")]]
[[Precondition("encode", "suffix_len <= 1 || num_v_heads == 0 || head_v_dim == 0 || num_v_heads <= value_dim / head_v_dim")]]
func DeltaNetChunkedOutputAndState<T: IModelFloat, O: IModelFloat>(
  Ptr<float, Access.Read> q_norm,
  Ptr<float, Access.Read> k_norm,
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> qk_scaled,
  Ptr<float, Access.Read> t_mat,
  Ptr<float, Access.Read> g,
  Ptr<float, Access.Read> beta,
  Ptr<float, Access.ReadWrite> state,
  Ptr<O, Access.ReadWrite> out,
  uint num_v_heads,
  uint num_k_heads,
  uint head_v_dim,
  uint key_dim,
  uint value_dim,
  uint suffix_len,
  [[Groups("if suffix_len == 0 { 0 } else { num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint hv_group,
  [[Groups("(head_v_dim / 32).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint value_tile_group,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
  [[Specialize]] uint VT,
  [[Specialize]] bool USE_MXU,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let chunks = suffix_len / CHUNKED_OUTPUT_CHUNK + (suffix_len % CHUNKED_OUTPUT_CHUNK != 0 ? 1u : 0u);
  let value_tiles = head_v_dim / CHUNKED_OUTPUT_VT;
  let heads_per_group = num_v_heads / num_k_heads;
  let key = thread_idx;
  var hv = hv_group;
  while (hv < num_v_heads) {
    let head = element_index(hv / heads_per_group, 128, 0);
    let key_head = element_index(hv / heads_per_group, 128, key);
    let g_head = element_pointer(g, element_index(hv, suffix_len, 0));
    var value_tile = value_tile_group;
    while (value_tile < value_tiles) {
      let value_base = value_tile * CHUNKED_OUTPUT_VT;
      for (uint v = 0; v < CHUNKED_OUTPUT_VT; v++) {
        let state_row = delta_net_state_row(state, hv, head_v_dim, value_base + v);
        chunked_state_tile[key * CHUNKED_OUTPUT_VT + v] = element_pointer(state_row, uint2(key, 0))[0];
      }
      GroupMemoryBarrierWithGroupSync();
      for (uint chunk = 0; chunk < chunks; chunk++) {
        let token_base = chunk * CHUNKED_OUTPUT_CHUNK;
        let valid_tokens = min(CHUNKED_OUTPUT_CHUNK, suffix_len - token_base);
        let chunk_head = byte_offset(element_index(chunk, num_v_heads, hv), CHUNKED_OUTPUT_CHUNK * CHUNKED_OUTPUT_CHUNK);
        float query_projection[CHUNKED_OUTPUT_ENTRIES];
        for (uint i = 0; i < CHUNKED_OUTPUT_ENTRIES; i++) {
          let entry = thread_idx + i * CHUNKED_OUTPUT_THREADS;
          let t = entry / CHUNKED_OUTPUT_VT;
          let v = entry % CHUNKED_OUTPUT_VT;
          var residual = 0.0;
          query_projection[i] = 0.0;
          if (t < valid_tokens) {
            let token = token_base + t;
            let k_row = element_pointer(element_pointer(k_norm, element_index(token, key_dim, 0)), head);
            let q_row = element_pointer(element_pointer(q_norm, element_index(token, key_dim, 0)), head);
            var key_projection = 0.0;
            for (uint j = 0; j < 128; j++) {
              let s = chunked_state_tile[j * CHUNKED_OUTPUT_VT + v];
              key_projection = ordered_sum(key_projection, exact_product(element_pointer(k_row, uint2(j, 0))[0], s));
              query_projection[i] = ordered_sum(query_projection[i], exact_product(element_pointer(q_row, uint2(j, 0))[0], s));
            }
            let values = delta_net_rows(delta_net_rows(in_proj, token, uint3(key_dim, value_dim, num_v_heads)), 1, uint3(key_dim, 0, 0));
            let value_input = element_pointer(values, element_index(hv, head_v_dim, value_base + v))[0].toFloat();
            let decayed = exact_product(delta_net_exp(element_pointer(g_head, uint2(token, 0))[0]), key_projection);
            let beta_t = element_pointer(beta, element_index(token, num_v_heads, hv))[0];
            residual = exact_product(beta_t, ordered_sum(value_input, -decayed));
          }
          chunked_scratch_tile[entry] = residual;
        }
        GroupMemoryBarrierWithGroupSync();

        float new_value[CHUNKED_OUTPUT_ENTRIES];
        for (uint i = 0; i < CHUNKED_OUTPUT_ENTRIES; i++) {
          let entry = thread_idx + i * CHUNKED_OUTPUT_THREADS;
          let t = entry / CHUNKED_OUTPUT_VT;
          let v = entry % CHUNKED_OUTPUT_VT;
          let t_row = element_pointer(t_mat, add_address(chunk_head, element_index(t, CHUNKED_OUTPUT_CHUNK, 0)));
          new_value[i] = 0.0;
          for (uint s = 0; s < t - t % CHUNKED_OUTPUT_TOKEN_TILE + CHUNKED_OUTPUT_TOKEN_TILE; s++) {
            let causal = t < valid_tokens ? element_pointer(t_row, uint2(s, 0))[0].toFloat() : 0.0;
            new_value[i] = ordered_sum(new_value[i], exact_product(causal, chunked_scratch_tile[s * CHUNKED_OUTPUT_VT + v]));
          }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint i = 0; i < CHUNKED_OUTPUT_ENTRIES; i++) {
          chunked_scratch_tile[thread_idx + i * CHUNKED_OUTPUT_THREADS] = new_value[i];
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint i = 0; i < CHUNKED_OUTPUT_ENTRIES; i++) {
          let entry = thread_idx + i * CHUNKED_OUTPUT_THREADS;
          let t = entry / CHUNKED_OUTPUT_VT;
          let v = entry % CHUNKED_OUTPUT_VT;
          if (t < valid_tokens) {
            let token = token_base + t;
            let qk_row = element_pointer(qk_scaled, add_address(chunk_head, element_index(t, CHUNKED_OUTPUT_CHUNK, 0)));
            var output = exact_product(query_projection[i], delta_net_exp(element_pointer(g_head, uint2(token, 0))[0]));
            for (uint s = 0; s < t - t % CHUNKED_OUTPUT_TOKEN_TILE + CHUNKED_OUTPUT_TOKEN_TILE; s++) {
              let qk = s < valid_tokens ? element_pointer(qk_row, uint2(s, 0))[0] : 0.0;
              output = ordered_sum(output, exact_product(qk, chunked_scratch_tile[s * CHUNKED_OUTPUT_VT + v]));
            }
            let out_row = element_pointer(out, element_index(token, value_dim, 0));
            element_pointer(out_row, element_index(hv, head_v_dim, value_base + v))[0] = O.fromFloat(output);
          }
        }

        let g_last = element_pointer(g_head, uint2(token_base + valid_tokens - 1, 0))[0];
        float update[CHUNKED_OUTPUT_VT];
        for (uint v = 0; v < CHUNKED_OUTPUT_VT; v++) {
          update[v] = 0.0;
        }
        for (uint s = 0; s < CHUNKED_OUTPUT_CHUNK; s++) {
          var scaled_key = 0.0;
          if (s < valid_tokens) {
            let token = token_base + s;
            let k_row = element_pointer(k_norm, element_index(token, key_dim, 0));
            let decay = delta_net_exp(ordered_sum(g_last, -element_pointer(g_head, uint2(token, 0))[0]));
            scaled_key = exact_product(element_pointer(k_row, key_head)[0], decay);
          }
          for (uint v = 0; v < CHUNKED_OUTPUT_VT; v++) {
            let new_value_s = s < valid_tokens ? chunked_scratch_tile[s * CHUNKED_OUTPUT_VT + v] : 0.0;
            update[v] = ordered_sum(update[v], exact_product(new_value_s, scaled_key));
          }
        }
        let state_decay = delta_net_exp(g_last);
        for (uint v = 0; v < CHUNKED_OUTPUT_VT; v++) {
          let index = key * CHUNKED_OUTPUT_VT + v;
          chunked_state_tile[index] = ordered_sum(exact_product(state_decay, chunked_state_tile[index]), update[v]);
        }
        GroupMemoryBarrierWithGroupSync();
      }
      for (uint v = 0; v < CHUNKED_OUTPUT_VT; v++) {
        let state_row = delta_net_state_row(state, hv, head_v_dim, value_base + v);
        element_pointer(state_row, uint2(key, 0))[0] = chunked_state_tile[key * CHUNKED_OUTPUT_VT + v];
      }
      if (value_tiles - value_tile <= groups.y) {
        break;
      }
      value_tile += groups.y;
    }
    if (num_v_heads - hv <= groups.x) {
      break;
    }
    hv += groups.x;
  }
}
