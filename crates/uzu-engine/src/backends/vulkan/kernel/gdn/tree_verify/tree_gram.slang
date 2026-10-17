import definitions;
import common.activation;
import common.ordered_dot;
import common.storage_address;
import gdn.prefill_prep;
import generated.trie;

// The CPU BuildTreeGram in its FP32 operation order, never fused, with hk = hv / floor(value_heads / k_heads). Each
// workgroup owns one (16-row block, v head) for every batch: its qkd rows, A tiles of pairs <= block / 2 (padding +0,
// later pairs untouched) and its identity-padded inverse block (z group 0 only) and, for a nonnegative h0 slot only, the
// kh0 rows of its 16-column dv tiles, staged through groupshared 64-wide k blocks in the same ascending order. Strides
// use the clamped NumWorkgroups, so every invocation reaches each barrier; addresses are termwise 64-bit.

static const uint TREE_GRAM_THREADS = 128;
static const uint TREE_GRAM_BLOCK = 16;
static const uint TREE_GRAM_K_BLOCK = 64;

groupshared float tree_gram_diag[TREE_GRAM_BLOCK * TREE_GRAM_BLOCK];
// k[row][d] with row stride 65 and h0[dv][d] transposed to [d][dv] with stride 17.
groupshared float tree_gram_k[TREE_GRAM_BLOCK * (TREE_GRAM_K_BLOCK + 1)];
groupshared float tree_gram_h0[TREE_GRAM_K_BLOCK * (TREE_GRAM_BLOCK + 1)];

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
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 { 0 } else { (if use_h0 { head_v_dim.div_ceil(16).max(1) } else { 1 }).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[2]) }")]] uint dv_tile_group,
  [[Threads("128")]] uint thread_idx,
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
        if (dv_tile_group == 0) {
          let qkd_head = element_pointer(qkd, byte_offset(byte_offset(head, tree_size), tree_size));
          var first_col = 0u;
          while (true) {
            for (uint entry = thread_idx; entry < rows * TREE_GRAM_BLOCK; entry += TREE_GRAM_THREADS) {
              let row = first + entry / TREE_GRAM_BLOCK;
              let col = first_col + entry % TREE_GRAM_BLOCK;
              if (col < tree_size) {
                let node = element_pointer(trie, element_index(b, tree_size, col))[0];
                var value = 0.0;
                if (row >= node.trie_start && row <= node.trie_end) {
                  let p_row = element_pointer(prefix, tree_gram_lane(b, tree_size, row, value_heads, hv))[0];
                  let q_row = element_pointer(q, byte_offset(tree_gram_lane(b, tree_size, row, k_heads, hk), head_k_dim));
                  let k_col = element_pointer(k, byte_offset(tree_gram_lane(b, tree_size, col, k_heads, hk), head_k_dim));
                  let qk = ordered_dot(q_row, k_col, head_k_dim);
                  let p_col = element_pointer(prefix, tree_gram_lane(b, tree_size, col, value_heads, hv))[0];
                  value = exact_product(exact_product(delta_net_exp(ordered_sum(p_row, -p_col)), scale), qk);
                }
                element_pointer(qkd_head, element_index(row, tree_size, col))[0] = value;
              }
            }
            if (tree_size - first_col <= TREE_GRAM_BLOCK) {
              break;
            }
            first_col += TREE_GRAM_BLOCK;
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
        }
        if (use_h0) {
          let h0_head = byte_offset(element_index(b, value_heads, hv), head_v_dim);
          let local_dv = thread_idx % TREE_GRAM_BLOCK;
          let local_row = thread_idx / TREE_GRAM_BLOCK;
          let dv_tiles = head_v_dim / TREE_GRAM_BLOCK + (head_v_dim % TREE_GRAM_BLOCK != 0 ? 1u : 0u);
          var dv_tile = dv_tile_group;
          while (dv_tile < dv_tiles) {
            let first_dv = dv_tile * TREE_GRAM_BLOCK;
            float sums[2] = { 0.0, 0.0 };
            var first_d = 0u;
            while (first_d < head_k_dim) {
              let depth = min(TREE_GRAM_K_BLOCK, head_k_dim - first_d);
              for (uint entry = thread_idx; entry < TREE_GRAM_BLOCK * TREE_GRAM_K_BLOCK; entry += TREE_GRAM_THREADS) {
                let line = entry / TREE_GRAM_K_BLOCK;
                let d = entry % TREE_GRAM_K_BLOCK;
                var k_value = 0.0;
                var h0_value = 0.0;
                if (d < depth) {
                  if (line < rows) {
                    let k_row = element_pointer(k, byte_offset(tree_gram_lane(b, tree_size, first + line, k_heads, hk), head_k_dim));
                    k_value = element_pointer(k_row, uint2(first_d + d, 0))[0].toFloat();
                  }
                  if (first_dv + line < head_v_dim) {
                    let h0_row = element_pointer(h0, byte_offset(add_address(h0_head, uint2(first_dv + line, 0)), head_k_dim));
                    h0_value = element_pointer(h0_row, uint2(first_d + d, 0))[0];
                  }
                }
                tree_gram_k[line * (TREE_GRAM_K_BLOCK + 1) + d] = k_value;
                tree_gram_h0[d * (TREE_GRAM_BLOCK + 1) + line] = h0_value;
              }
              GroupMemoryBarrierWithGroupSync();
              for (uint half = 0; half < 2; half++) {
                for (uint d = 0; d < depth; d++) {
                  let product = exact_product(
                    tree_gram_k[(local_row + half * (TREE_GRAM_THREADS / TREE_GRAM_BLOCK)) * (TREE_GRAM_K_BLOCK + 1) + d],
                    tree_gram_h0[d * (TREE_GRAM_BLOCK + 1) + local_dv]
                  );
                  sums[half] = ordered_sum(sums[half], product);
                }
              }
              GroupMemoryBarrierWithGroupSync();
              if (depth == head_k_dim - first_d) {
                break;
              }
              first_d += TREE_GRAM_K_BLOCK;
            }
            let dv = first_dv + local_dv;
            for (uint half = 0; half < 2; half++) {
              let row_index = local_row + half * (TREE_GRAM_THREADS / TREE_GRAM_BLOCK);
              if (row_index < rows && dv < head_v_dim) {
                let kh0_lane = tree_gram_lane(b, tree_size, first + row_index, value_heads, hv);
                let kh0_row = element_pointer(kh0, byte_offset(kh0_lane, head_v_dim));
                element_pointer(kh0_row, uint2(dv, 0))[0] = sums[half];
              }
            }
            if (dv_tiles - dv_tile <= groups.z) {
              break;
            }
            dv_tile += groups.z;
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
