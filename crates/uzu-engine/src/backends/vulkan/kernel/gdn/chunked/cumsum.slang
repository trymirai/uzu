import definitions;
import common.activation;
import common.storage_address;

// The Metal DeltaNetChunkedCumsum: g[hv, t] is the inclusive prefix of log_decay[t, hv] within t's CHUNK_SIZE chunk. As
// over Metal's 32 lanes, each 32-token window's prefix runs from +0 in token order and is added to the carry, the sum of
// the earlier windows' totals; each sum is one IEEE FP32 operation. One invocation owns one (chunk, v head), 128 chunks
// per workgroup. Workgroups stride over chunks and heads by the actual NumWorkgroups, which the host clamps to the
// device limits, advancing only while more than one step remains.

static const uint CHUNKED_CUMSUM_THREADS = 128;
static const uint CHUNKED_CUMSUM_WINDOW = 32;

[[Kernel]]
[[Precondition("new", "CHUNK_SIZE == 32 || CHUNK_SIZE == 64")]]
func DeltaNetChunkedCumsum(
  Ptr<float, Access.Read> log_decay,
  Ptr<float, Access.ReadWrite> g_out,
  uint num_v_heads,
  uint suffix_len,
  [[Groups("if suffix_len == 0 { 0 } else { suffix_len.div_ceil(8192).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint chunk_group,
  [[Groups("num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint hv_group,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint CHUNK_SIZE,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let chunks = suffix_len / CHUNK_SIZE + (suffix_len % CHUNK_SIZE != 0 ? 1u : 0u);
  let stride = groups.x * CHUNKED_CUMSUM_THREADS;
  var hv = hv_group;
  while (hv < num_v_heads) {
    var chunk = chunk_group * CHUNKED_CUMSUM_THREADS + thread_idx;
    while (chunk < chunks) {
      let token_base = chunk * CHUNK_SIZE;
      let valid_tokens = min(CHUNK_SIZE, suffix_len - token_base);
      let g_row = element_pointer(g_out, element_index(hv, suffix_len, token_base));
      var carry = 0.0;
      for (uint window = 0; window < valid_tokens; window += CHUNKED_CUMSUM_WINDOW) {
        var prefix = 0.0;
        for (uint offset = window; offset < min(window + CHUNKED_CUMSUM_WINDOW, valid_tokens); offset++) {
          prefix = ordered_sum(prefix, element_pointer(log_decay, element_index(token_base + offset, num_v_heads, hv))[0]);
          element_pointer(g_row, uint2(offset, 0))[0] = ordered_sum(carry, prefix);
        }
        carry = ordered_sum(carry, prefix);
      }
      if (chunks - chunk <= stride) {
        break;
      }
      chunk += stride;
    }
    if (num_v_heads - hv <= groups.y) {
      break;
    }
    hv += groups.y;
  }
}
