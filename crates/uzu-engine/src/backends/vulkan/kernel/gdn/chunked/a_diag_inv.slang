import definitions;
import common.activation;
import common.storage_address;
import gdn.prefill_prep;

// The Metal DeltaNetChunkedADiagInv, with hk = hv / floor(H / G). For each 16-row block of a chunk, A[r, c] =
// (beta[r] e^(g_r - g_c)) kk[r, c] where c < r < valid tokens, otherwise +0, fills the 16 x 32 column-pair tiles
// a_packed[chunk, hv, block, pair] through the diagonal pair (later pairs untouched), and the column of
// a_inv[chunk, hv, block] = (I + A_diag)^-1 for c < 16 is the identity column with, for c < r < the block's valid rows in
// ascending r, row r = -(sum over p < r of A_diag[r, p] column[p]) from +0: every product is formed, also with +0
// entries, as Metal's invert_lower_triangular_block. Each product and sum is one IEEE FP32 operation, never fused. One
// workgroup owns one (chunk, v head); strides use the clamped NumWorkgroups, so every invocation reaches each barrier.
// Element offsets are termwise 64-bit.

static const uint CHUNKED_A_THREADS = 128;
static const uint CHUNKED_A_BLOCK = 16;

groupshared float chunked_a_diag[CHUNKED_A_BLOCK * CHUNKED_A_BLOCK];

[[Kernel]]
[[Precondition("new", "CHUNK_SIZE == 32 || CHUNK_SIZE == 64")]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "num_v_heads == 0 || num_v_heads >= num_k_heads")]]
func DeltaNetChunkedADiagInv(
  Ptr<float, Access.Read> kk,
  Ptr<float, Access.Read> beta,
  Ptr<float, Access.Read> g,
  Ptr<float, Access.ReadWrite> a_packed,
  Ptr<float, Access.ReadWrite> a_inv,
  uint num_v_heads,
  uint num_k_heads,
  uint suffix_len,
  [[Groups("if suffix_len == 0 { 0 } else { suffix_len.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint chunk_group,
  [[Groups("num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint hv_group,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint CHUNK_SIZE,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let chunks = suffix_len / CHUNK_SIZE + (suffix_len % CHUNK_SIZE != 0 ? 1u : 0u);
  let blocks = CHUNK_SIZE / CHUNKED_A_BLOCK;
  let pairs = blocks / 2 + blocks % 2;
  let heads_per_group = num_v_heads / num_k_heads;
  var hv = hv_group;
  while (hv < num_v_heads) {
    let hk = hv / heads_per_group;
    var chunk = chunk_group;
    while (chunk < chunks) {
      let token_base = chunk * CHUNK_SIZE;
      let valid_tokens = min(CHUNK_SIZE, suffix_len - token_base);
      let head = element_index(chunk, num_v_heads, hv);
      let g_head = element_pointer(g, element_index(hv, suffix_len, token_base));
      let kk_head = element_pointer(kk, byte_offset(element_index(chunk, num_k_heads, hk), CHUNK_SIZE * CHUNK_SIZE));
      let tiles = element_pointer(a_packed, byte_offset(byte_offset(byte_offset(head, blocks), pairs), 512));
      let inverses = element_pointer(a_inv, byte_offset(byte_offset(head, blocks), 256));
      for (uint block = 0; block < blocks; block++) {
        let row_base = block * CHUNKED_A_BLOCK;
        let diag_col = (block % 2) * CHUNKED_A_BLOCK;
        for (uint pair = 0; pair <= block / 2; pair++) {
          let tile = element_pointer(tiles, byte_offset(element_index(block, pairs, pair), 512));
          for (uint entry = thread_idx; entry < 2 * CHUNKED_A_BLOCK * CHUNKED_A_BLOCK; entry += CHUNKED_A_THREADS) {
            let local_row = entry / (2 * CHUNKED_A_BLOCK);
            let local_col = entry % (2 * CHUNKED_A_BLOCK);
            let row = row_base + local_row;
            let col = pair * 2 * CHUNKED_A_BLOCK + local_col;
            var value = 0.0;
            if (col < row && row < valid_tokens) {
              let g_row = element_pointer(g_head, uint2(row, 0))[0];
              let e = delta_net_exp(ordered_sum(g_row, -element_pointer(g_head, uint2(col, 0))[0]));
              let beta_row = element_pointer(beta, element_index(token_base + row, num_v_heads, hv))[0];
              value = exact_product(exact_product(beta_row, e), element_pointer(kk_head, element_index(row, CHUNK_SIZE, col))[0]);
            }
            element_pointer(tile, uint2(entry, 0))[0] = value;
            if (pair == block / 2 && local_col >= diag_col && local_col < diag_col + CHUNKED_A_BLOCK) {
              chunked_a_diag[local_row * CHUNKED_A_BLOCK + local_col - diag_col] = value;
            }
          }
        }
        GroupMemoryBarrierWithGroupSync();
        if (thread_idx < CHUNKED_A_BLOCK) {
          let c = thread_idx;
          let block_rows = row_base < valid_tokens ? min(CHUNKED_A_BLOCK, valid_tokens - row_base) : 0u;
          float column[CHUNKED_A_BLOCK];
          for (uint r = 0; r < CHUNKED_A_BLOCK; r++) {
            column[r] = r == c ? 1.0 : 0.0;
          }
          for (uint r = c + 1; r < block_rows; r++) {
            var s = 0.0;
            for (uint p = 0; p < r; p++) {
              s = ordered_sum(s, exact_product(chunked_a_diag[r * CHUNKED_A_BLOCK + p], column[p]));
            }
            column[r] = -s;
          }
          let inverse = element_pointer(inverses, byte_offset(uint2(block, 0), 256));
          for (uint r = 0; r < CHUNKED_A_BLOCK; r++) {
            element_pointer(inverse, element_index(r, CHUNKED_A_BLOCK, c))[0] = column[r];
          }
        }
        GroupMemoryBarrierWithGroupSync();
      }
      if (chunks - chunk <= groups.x) {
        break;
      }
      chunk += groups.x;
    }
    if (num_v_heads - hv <= groups.y) {
      break;
    }
    hv += groups.y;
  }
}
