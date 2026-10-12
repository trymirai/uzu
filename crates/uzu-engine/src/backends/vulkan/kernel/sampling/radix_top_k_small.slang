import definitions;
import common.ranking;
import common.storage_address;

groupshared uint radix_top_k_histogram[256];
groupshared uint2 radix_top_k_keys[512];
groupshared uint2 radix_top_k_prefix;
groupshared uint2 radix_top_k_mask;
groupshared uint radix_top_k_rank;
groupshared uint radix_top_k_done;
groupshared uint radix_top_k_count;

// The unique descending key of a column: x the f32::total_cmp key of its expanded BF16, y the inverted column, so a
// lower column ranks higher on equal scores. Every valid x is at least 0x0000ffff, so uint2(0, 0) is below all keys.
uint2 radix_top_k_key(Ptr<bf16, Access.Read> row_input, uint column) {
  let bits = asuint(element_pointer(row_input, uint2(column, 0))[0].toFloat());
  return uint2(float_order_key(bits), ~column);
}

bool radix_top_k_above(uint2 a, uint2 b) {
  return a.x > b.x || (a.x == b.x && a.y > b.y);
}

// One workgroup per row finds the k-th largest key by 8-bit radix digits: two over the score key's upper half, whose
// lower half the sign then fixes (zeros above positive values, ones above negative ones), and four over the inverted
// column. A digit whose bucket holds exactly the remaining rank ends the search, since its whole bucket is selected.
// Every key at or above the threshold is then one of exactly k, collected and bitonic-sorted in shared memory.
[[Kernel]]
[[Precondition("new", "columns > 0")]]
[[Precondition("encode", "rows > 0 && k > 0 && k <= 512 && k <= columns")]]
func RadixTopKSmall(
  Ptr<bf16, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output_ids,
  Ptr<float, Access.ReadWrite> output_scores,
  uint rows,
  uint k,
  [[Specialize]] uint columns,
  [[Groups("rows.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0])")]] uint group,
  [[Threads("256")]] uint tid,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  var row = group;
  while (row < rows) {
    let row_input = element_pointer(input, element_index(row, columns, 0));
    radix_top_k_keys[tid] = uint2(0, 0);
    radix_top_k_keys[tid + 256] = uint2(0, 0);
    if (tid == 0) {
      radix_top_k_prefix = uint2(0, 0);
      radix_top_k_mask = uint2(0, 0);
      radix_top_k_rank = k - 1;
      radix_top_k_done = 0;
      radix_top_k_count = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint pass = 0; pass < 6; pass++) {
      if (radix_top_k_done != 0) { break; }
      radix_top_k_histogram[tid] = 0;
      GroupMemoryBarrierWithGroupSync();
      let prefix = radix_top_k_prefix;
      let mask = radix_top_k_mask;
      let shift = pass < 2 ? 24 - 8 * pass : 40 - 8 * pass;
      for (uint column = tid; column < columns; column += 256) {
        let key = radix_top_k_key(row_input, column);
        if (all((key & mask) == prefix)) {
          InterlockedAdd(radix_top_k_histogram[((pass < 2 ? key.x : key.y) >> shift) & 0xffu], 1u);
        }
        if (columns - column <= 256) { break; }
      }
      GroupMemoryBarrierWithGroupSync();
      if (tid == 0) {
        var rank = radix_top_k_rank;
        var bucket = 255u;
        while (rank >= radix_top_k_histogram[bucket]) {
          rank -= radix_top_k_histogram[bucket];
          bucket--;
        }
        var next_prefix = prefix;
        var next_mask = mask;
        if (pass < 2) {
          next_prefix.x |= bucket << shift;
          next_mask.x |= 0xffu << shift;
          if (pass == 1) {
            next_prefix.x |= (next_prefix.x & 0x80000000u) != 0 ? 0u : 0xffffu;
            next_mask.x = 0xffffffffu;
          }
        } else {
          next_prefix.y |= bucket << shift;
          next_mask.y |= 0xffu << shift;
        }
        radix_top_k_prefix = next_prefix;
        radix_top_k_mask = next_mask;
        radix_top_k_rank = rank;
        radix_top_k_done = radix_top_k_histogram[bucket] == rank + 1 ? 1u : 0u;
      }
      GroupMemoryBarrierWithGroupSync();
    }

    let threshold = radix_top_k_prefix;
    for (uint column = tid; column < columns; column += 256) {
      let key = radix_top_k_key(row_input, column);
      if (!radix_top_k_above(threshold, key)) {
        uint slot;
        InterlockedAdd(radix_top_k_count, 1u, slot);
        radix_top_k_keys[slot] = key;
      }
      if (columns - column <= 256) { break; }
    }
    GroupMemoryBarrierWithGroupSync();

    uint span = 1;
    while (span < k) { span <<= 1; }
    for (uint size = 2; size <= span; size <<= 1) {
      for (uint distance = size >> 1; distance > 0; distance >>= 1) {
        let lower = (tid & (distance - 1)) | ((tid & ~(distance - 1)) << 1);
        let higher = lower | distance;
        if (higher < span) {
          let a = radix_top_k_keys[lower];
          let b = radix_top_k_keys[higher];
          let swap = radix_top_k_above(b, a) == ((lower & size) == 0);
          radix_top_k_keys[lower] = swap ? b : a;
          radix_top_k_keys[higher] = swap ? a : b;
        }
        GroupMemoryBarrierWithGroupSync();
      }
    }

    for (uint index = tid; index < k; index += 256) {
      let key = radix_top_k_keys[index];
      let destination = element_index(row, k, index);
      element_pointer(output_ids, destination)[0] = ~key.y;
      element_pointer(output_scores, destination)[0] = asfloat(float_order_bits(key.x));
    }
    // Every invocation has read its keys before the next row clears them.
    GroupMemoryBarrierWithGroupSync();
    if (groups.x > rows - 1 - row) { break; }
    row += groups.x;
  }
}
