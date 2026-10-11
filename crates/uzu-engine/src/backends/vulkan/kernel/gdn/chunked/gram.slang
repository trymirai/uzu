import definitions;
import common.activation;
import common.ordered_dot;
import common.storage_address;
import gdn.prefill_prep;

// The Metal DeltaNetChunkedGram over [chunk, 16-row, 32-column] tiles, with the q/k rows of token t and k head hk at
// t key_dim + hk 128. A tile with valid rows and columns, whose first column lies before the end of its valid rows, is
// written whole: kk[r, c] = k_r . k_c and, for each grouped v head hv = hk floor(H / G) + group,
// qk_scaled[r, c] = (q_r . k_c) e^(g_r - g_c) where c <= r < valid tokens, otherwise +0. A row or column past the suffix
// enters its dots as +0 and is never read. Other tiles stay untouched, as in Metal. Dots run from +0 over the 128 key
// elements in ascending order, and each product and sum is one IEEE FP32 operation. One workgroup owns one
// (chunk, k head); workgroups stride over chunks and heads by the actual NumWorkgroups, which the host clamps to the
// device limits, advancing only while more than one step remains. Element offsets are termwise 64-bit.

static const uint CHUNKED_GRAM_THREADS = 128;
static const uint CHUNKED_GRAM_ROW_TILE = 16;
static const uint CHUNKED_GRAM_COL_TILE = 32;

// x . y over 128 elements, an absent row taken as +0 in every product: NaN where the other row is not finite.
float chunked_gram_dot(Ptr<float, Access.Read> x, bool x_valid, Ptr<float, Access.Read> y, bool y_valid) {
  if (x_valid && y_valid) {
    return ordered_dot(x, y, 128);
  }
  var sum = 0.0;
  for (uint d = 0; d < 128; d++) {
    let x_d = x_valid ? element_pointer(x, uint2(d, 0))[0] : 0.0;
    let y_d = y_valid ? element_pointer(y, uint2(d, 0))[0] : 0.0;
    sum = ordered_sum(sum, exact_product(x_d, y_d));
  }
  return sum;
}

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
  [[Groups("if suffix_len == 0 { 0 } else { suffix_len.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint chunk_group,
  [[Groups("num_k_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint hk_group,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
  [[Specialize]] uint CHUNK_SIZE,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let chunks = suffix_len / CHUNK_SIZE + (suffix_len % CHUNK_SIZE != 0 ? 1u : 0u);
  let heads_per_group = num_v_heads / num_k_heads;
  var hk = hk_group;
  while (hk < num_k_heads) {
    let key_head = element_index(hk, 128, 0);
    var chunk = chunk_group;
    while (chunk < chunks) {
      let token_base = chunk * CHUNK_SIZE;
      let valid_tokens = min(CHUNK_SIZE, suffix_len - token_base);
      let kk_head = element_pointer(kk_out, byte_offset(element_index(chunk, num_k_heads, hk), CHUNK_SIZE * CHUNK_SIZE));
      for (uint entry = thread_idx; entry < CHUNK_SIZE * CHUNK_SIZE; entry += CHUNKED_GRAM_THREADS) {
        let row = entry / CHUNK_SIZE;
        let col = entry % CHUNK_SIZE;
        let row_base = row - row % CHUNKED_GRAM_ROW_TILE;
        let col_base = col - col % CHUNKED_GRAM_COL_TILE;
        let valid_rows = row_base < valid_tokens ? min(CHUNKED_GRAM_ROW_TILE, valid_tokens - row_base) : 0u;
        if (valid_rows == 0 || col_base >= valid_tokens || col_base >= row_base + valid_rows) {
          continue;
        }
        let row_valid = row < valid_tokens;
        let col_valid = col < valid_tokens;
        var k_row = k_norm;
        var q_row = q_norm;
        var k_col = k_norm;
        if (row_valid) {
          k_row = element_pointer(element_pointer(k_norm, element_index(token_base + row, key_dim, 0)), key_head);
          q_row = element_pointer(element_pointer(q_norm, element_index(token_base + row, key_dim, 0)), key_head);
        }
        if (col_valid) {
          k_col = element_pointer(element_pointer(k_norm, element_index(token_base + col, key_dim, 0)), key_head);
        }
        element_pointer(kk_head, uint2(entry, 0))[0] = chunked_gram_dot(k_row, row_valid, k_col, col_valid);
        let qk = chunked_gram_dot(q_row, row_valid, k_col, col_valid);
        for (uint group = 0; group < heads_per_group; group++) {
          let hv = hk * heads_per_group + group;
          var value = 0.0;
          if (col_valid && row_valid && col <= row) {
            let g_head = element_pointer(g, element_index(hv, suffix_len, token_base));
            let g_row = element_pointer(g_head, uint2(row, 0))[0];
            let g_col = element_pointer(g_head, uint2(col, 0))[0];
            value = exact_product(qk, delta_net_exp(ordered_sum(g_row, -g_col)));
          }
          let qk_head = byte_offset(element_index(chunk, num_v_heads, hv), CHUNK_SIZE * CHUNK_SIZE);
          element_pointer(element_pointer(qk_scaled_out, qk_head), uint2(entry, 0))[0] = value;
        }
      }
      if (chunks - chunk <= groups.x) {
        break;
      }
      chunk += groups.x;
    }
    if (num_k_heads - hk <= groups.y) {
      break;
    }
    hk += groups.y;
  }
}
