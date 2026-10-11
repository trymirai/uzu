import definitions;
import common.activation;
import common.storage_address;

// Dense T = (I + A)^-1 over a chunk in FP32, by block forward substitution in BV-wide
// column slices. For each 16-row block in order, rhs = the identity slice, then rhs[r, c] += -A[row, j] T[j, c] over
// the earlier columns j in ascending order, then T[row, c] = sum over p of inverse[r, p] rhs[p, c] from +0.
// Each product and sum is one IEEE FP32 operation, never fused. Every row is
// written, past the suffix too, as in Metal. One workgroup owns one (chunk, v head) and walks its slices; strides use
// the clamped NumWorkgroups, so every invocation reaches each barrier. Element offsets are termwise 64-bit.

static const uint CHUNKED_T_THREADS = 128;
static const uint CHUNKED_T_BLOCK = 16;

// The slice's T rows [CHUNK_SIZE, BV] and one block's rhs [16, BV], at the largest variants.
groupshared float chunked_t_rows[64 * 32];
groupshared float chunked_t_rhs[CHUNKED_T_BLOCK * 32];

[[Kernel]]
[[Precondition("new", "CHUNK_SIZE == 32 || CHUNK_SIZE == 64")]]
[[Precondition("new", "BV == 16 || BV == 32")]]
func DeltaNetChunkedCausalInv(
  Ptr<float, Access.Read> a_packed,
  Ptr<float, Access.Read> a_inv,
  Ptr<float, Access.ReadWrite> t_out,
  uint num_v_heads,
  uint suffix_len,
  [[Groups("if suffix_len == 0 { 0 } else { suffix_len.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint chunk_group,
  [[Groups("num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint hv_group,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint CHUNK_SIZE,
  [[Specialize]] uint BV,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let chunks = suffix_len / CHUNK_SIZE + (suffix_len % CHUNK_SIZE != 0 ? 1u : 0u);
  let blocks = CHUNK_SIZE / CHUNKED_T_BLOCK;
  let pairs = blocks / 2 + blocks % 2;
  var hv = hv_group;
  while (hv < num_v_heads) {
    var chunk = chunk_group;
    while (chunk < chunks) {
      let head = element_index(chunk, num_v_heads, hv);
      let tiles = element_pointer(a_packed, byte_offset(byte_offset(byte_offset(head, blocks), pairs), 512));
      let inverses = element_pointer(a_inv, byte_offset(byte_offset(head, blocks), 256));
      let t_head = element_pointer(t_out, byte_offset(head, CHUNK_SIZE * CHUNK_SIZE));
      for (uint slice_base = 0; slice_base < CHUNK_SIZE; slice_base += BV) {
        for (uint block = 0; block < blocks; block++) {
          let row_base = block * CHUNKED_T_BLOCK;
          for (uint entry = thread_idx; entry < CHUNKED_T_BLOCK * BV; entry += CHUNKED_T_THREADS) {
            let r = entry / BV;
            let c = entry % BV;
            var rhs = row_base + r == slice_base + c ? 1.0 : 0.0;
            for (uint j = 0; j < row_base; j++) {
              let tile = element_pointer(tiles, byte_offset(element_index(block, pairs, j / (2 * CHUNKED_T_BLOCK)), 512));
              let a = element_pointer(tile, element_index(r, 2 * CHUNKED_T_BLOCK, j % (2 * CHUNKED_T_BLOCK)))[0];
              rhs = ordered_sum(rhs, exact_product(-a, chunked_t_rows[j * BV + c]));
            }
            chunked_t_rhs[entry] = rhs;
          }
          GroupMemoryBarrierWithGroupSync();
          let inverse = element_pointer(inverses, byte_offset(uint2(block, 0), 256));
          for (uint entry = thread_idx; entry < CHUNKED_T_BLOCK * BV; entry += CHUNKED_T_THREADS) {
            let r = entry / BV;
            let c = entry % BV;
            var t = 0.0;
            for (uint p = 0; p < CHUNKED_T_BLOCK; p++) {
              let inverse_rp = element_pointer(inverse, element_index(r, CHUNKED_T_BLOCK, p))[0];
              t = ordered_sum(t, exact_product(inverse_rp, chunked_t_rhs[p * BV + c]));
            }
            element_pointer(t_head, element_index(row_base + r, CHUNK_SIZE, slice_base + c))[0] = t;
            chunked_t_rows[(row_base + r) * BV + c] = t;
          }
          GroupMemoryBarrierWithGroupSync();
        }
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
