import definitions;
import common.activation;
import common.ordered_dot;
import common.storage_address;
import gdn.prefill_prep;
import generated.trie;

// The CPU BuildTreeGram in its FP32 operation order, never fused, with hk = hv / floor(value_heads / k_heads). Each
// workgroup owns one (16-row block, v head) for every batch: its qkd rows, A tiles of pairs <= block / 2 (padding +0,
// later pairs untouched), its identity-padded inverse block and, for a nonnegative h0 slot only, its kh0 rows.
// Strides use the clamped NumWorkgroups, so every invocation reaches each barrier; addresses are termwise 64-bit.

static const uint TREE_GRAM_THREADS = 64;
static const uint TREE_GRAM_BLOCK = 16;

groupshared float tree_gram_diag[TREE_GRAM_BLOCK * TREE_GRAM_BLOCK];

// The element (b, row, head) of a [batch, tree_size, heads] lane array.
uint2 tree_gram_lane(uint b, uint tree_size, uint row, uint heads, uint head) {
  return add_address(byte_offset(element_index(b, tree_size, row), heads), uint2(head, 0));
}

[[Kernel]] [[Public]]
[[Precondition("new", "!USE_MXU")]]
[[Precondition("encode", "k_heads != 0 && (batch_size == 0 || value_heads == 0 || value_heads >= k_heads)")]]
func BuildTreeGram<T: IModelFloat>(
  Ptr<T, Access.Read> q,
  Ptr<T, Access.Read> k,
  Ptr<TrieNode, Access.Read> trie,
  Ptr<float, Access.Read> prefix,
  Ptr<float, Access.Read> beta,
  [[Optional("use_h0")]] Ptr<float, Access.Read> h0,
  [[Optional("use_h0")]] Ptr<int, Access.Read> h0_idx,
  Ptr<float, Access.ReadWrite> a_packed,
  Ptr<float, Access.ReadWrite> qkd,
  Ptr<float, Access.ReadWrite> a_inv,
  [[Optional("use_h0")]] Ptr<float, Access.ReadWrite> kh0,
  float scale,
  uint batch_size,
  uint tree_size,
  uint k_heads,
  uint value_heads,
  uint head_k_dim,
  uint head_v_dim,
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 { 0 } else { tree_size.div_ceil(16).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint block_group,
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 { 0 } else { value_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1]) }")]] uint head_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool USE_MXU,
  [[Specialize]] bool use_h0,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let blocks = tree_size / TREE_GRAM_BLOCK + (tree_size % TREE_GRAM_BLOCK != 0 ? 1u : 0u);
  let pairs = blocks / 2 + blocks % 2;
  let heads_per_group = value_heads / k_heads;
  var hv = head_group;
  while (hv < value_heads) {
    let hk = hv / heads_per_group;
    var block = block_group;
    while (block < blocks) {
      let first = block * TREE_GRAM_BLOCK;
      let rows = min(TREE_GRAM_BLOCK, tree_size - first);
      for (uint b = 0; b < batch_size; b++) {
        let head = element_index(b, value_heads, hv);
        let qkd_head = element_pointer(qkd, byte_offset(byte_offset(head, tree_size), tree_size));
        for (uint local_row = 0; local_row < rows; local_row++) {
          let row = first + local_row;
          let p_row = element_pointer(prefix, tree_gram_lane(b, tree_size, row, value_heads, hv))[0];
          let q_row = element_pointer(q, byte_offset(tree_gram_lane(b, tree_size, row, k_heads, hk), head_k_dim));
          var col = thread_idx;
          while (col < tree_size) {
            let node = element_pointer(trie, element_index(b, tree_size, col))[0];
            var value = 0.0;
            if (row >= node.trie_start && row <= node.trie_end) {
              let k_col = element_pointer(k, byte_offset(tree_gram_lane(b, tree_size, col, k_heads, hk), head_k_dim));
              let qk = ordered_dot(q_row, k_col, head_k_dim);
              let p_col = element_pointer(prefix, tree_gram_lane(b, tree_size, col, value_heads, hv))[0];
              value = exact_product(exact_product(delta_net_exp(ordered_sum(p_row, -p_col)), scale), qk);
            }
            element_pointer(qkd_head, element_index(row, tree_size, col))[0] = value;
            if (tree_size - col <= TREE_GRAM_THREADS) {
              break;
            }
            col += TREE_GRAM_THREADS;
          }
        }
        let tiles = element_pointer(a_packed, byte_offset(byte_offset(byte_offset(head, blocks), pairs), 512));
        let diag_col = (block % 2) * TREE_GRAM_BLOCK;
        for (uint pair = 0; pair <= block / 2; pair++) {
          let tile = element_pointer(tiles, byte_offset(element_index(block, pairs, pair), 512));
          for (uint entry = thread_idx; entry < 2 * TREE_GRAM_BLOCK * TREE_GRAM_BLOCK; entry += TREE_GRAM_THREADS) {
            let local_row = entry / (2 * TREE_GRAM_BLOCK);
            let local_col = entry % (2 * TREE_GRAM_BLOCK);
            let row = first + local_row;
            let col = pair * 2 * TREE_GRAM_BLOCK + local_col;
            var value = 0.0;
            if (row != col && row < tree_size && col < tree_size) {
              let node = element_pointer(trie, element_index(b, tree_size, col))[0];
              if (row >= node.trie_start && row <= node.trie_end) {
                let row_lane = tree_gram_lane(b, tree_size, row, value_heads, hv);
                let k_row = element_pointer(k, byte_offset(tree_gram_lane(b, tree_size, row, k_heads, hk), head_k_dim));
                let k_col = element_pointer(k, byte_offset(tree_gram_lane(b, tree_size, col, k_heads, hk), head_k_dim));
                let kk = ordered_dot(k_row, k_col, head_k_dim);
                let p_col = element_pointer(prefix, tree_gram_lane(b, tree_size, col, value_heads, hv))[0];
                let e = delta_net_exp(ordered_sum(element_pointer(prefix, row_lane)[0], -p_col));
                value = exact_product(exact_product(element_pointer(beta, row_lane)[0], e), kk);
              }
            }
            element_pointer(tile, element_index(local_row, 2 * TREE_GRAM_BLOCK, local_col))[0] = value;
            if (pair == block / 2 && local_col >= diag_col && local_col < diag_col + TREE_GRAM_BLOCK) {
              tree_gram_diag[local_row * TREE_GRAM_BLOCK + local_col - diag_col] = value;
            }
          }
        }
        GroupMemoryBarrierWithGroupSync();
        if (thread_idx < TREE_GRAM_BLOCK) {
          let c = thread_idx;
          float column[TREE_GRAM_BLOCK];
          for (uint r = 0; r < TREE_GRAM_BLOCK; r++) {
            column[r] = r == c ? 1.0 : 0.0;
          }
          for (uint r = c + 1; r < rows; r++) {
            var s = 0.0;
            for (uint p = c; p < r; p++) {
              s = ordered_sum(s, exact_product(tree_gram_diag[r * TREE_GRAM_BLOCK + p], column[p]));
            }
            column[r] = -s;
          }
          let inverses = element_pointer(a_inv, byte_offset(byte_offset(head, blocks), 256));
          let inverse = element_pointer(inverses, byte_offset(uint2(block, 0), 256));
          for (uint r = 0; r < TREE_GRAM_BLOCK; r++) {
            element_pointer(inverse, element_index(r, TREE_GRAM_BLOCK, c))[0] = column[r];
          }
        }
        if (use_h0) {
          let slot = element_pointer(h0_idx, uint2(b, 0))[0];
          if (slot >= 0) {
            let h0_head = byte_offset(element_index(uint(slot), value_heads, hv), head_v_dim);
            for (uint local_row = 0; local_row < rows; local_row++) {
              let row = first + local_row;
              let k_row = element_pointer(k, byte_offset(tree_gram_lane(b, tree_size, row, k_heads, hk), head_k_dim));
              let kh0_lane = tree_gram_lane(b, tree_size, row, value_heads, hv);
              let kh0_row = element_pointer(kh0, byte_offset(kh0_lane, head_v_dim));
              var dv = thread_idx;
              while (dv < head_v_dim) {
                let h0_row = element_pointer(h0, byte_offset(add_address(h0_head, uint2(dv, 0)), head_k_dim));
                element_pointer(kh0_row, uint2(dv, 0))[0] = ordered_dot(k_row, h0_row, head_k_dim);
                if (head_v_dim - dv <= TREE_GRAM_THREADS) {
                  break;
                }
                dv += TREE_GRAM_THREADS;
              }
            }
          }
        }
        GroupMemoryBarrierWithGroupSync();
      }
      if (blocks - block <= groups.x) {
        break;
      }
      block += groups.x;
    }
    if (value_heads - hv <= groups.y) {
      break;
    }
    hv += groups.y;
  }
}
