import definitions;
import common.ranking;
import common.workgroup_reduction;
import generated.radix_top_k;

// The k largest of each row by 8-bit radix digits over a unique descending key, spread over `partitions` workgroups per
// row so the reads use the whole GPU. A pass dispatch counts its partition's keys under the radix prefix into the row's
// histogram of that pass. Workgroups never wait on each other: each replays the earlier passes' digit choices from
// their histograms, which the dispatch boundaries make visible. Collect then appends the k keys at or above the final
// prefix, and Sort orders them.

static const uint RADIX_TOP_K_THREADS = 256;
static const uint RADIX_TOP_K_SCORE_PASSES = (bf16.FRACTION_BITS + 16) / 8;

groupshared uint radix_top_k_histogram[RADIX_TOP_K_BUCKETS];
groupshared uint2 radix_top_k_keys[512];
groupshared uint2 radix_top_k_prefix;
groupshared uint2 radix_top_k_mask;
groupshared uint radix_top_k_rank;
groupshared uint radix_top_k_done;

// The unique descending key of a column: x the f32::total_cmp key of its widened BF16, y the inverted column, so a lower
// column ranks higher on equal scores. Every valid x is nonzero, so uint2(0, 0) is below all keys.
uint2 radix_top_k_key(Ptr<bf16, Access.Read> input, uint row_offset, uint column) {
  return uint2(float_order_key(score_bits(input[row_offset + column])), ~column);
}

bool radix_top_k_above(uint2 a, uint2 b) {
  return a.x > b.x || (a.x == b.x && a.y > b.y);
}

// The bit shift of `pass`'s digit within its half of the key.
uint radix_top_k_shift(uint pass) {
  return 24 - 8 * (pass < RADIX_TOP_K_SCORE_PASSES ? pass : pass - RADIX_TOP_K_SCORE_PASSES);
}

// The radix prefix, mask and remaining rank of the k-th largest key after `passes` passes, chosen from their histograms;
// `radix_top_k_done` once a bucket holds exactly the remaining keys. Invocation i in subgroup_invocation order takes
// bucket 255 - i, so the keys of the higher buckets precede its own.
void radix_top_k_replay(Ptr<uint, Access.ReadWrite> histograms, uint row, uint k, uint passes) {
  static_assert(RADIX_TOP_K_SCORE_PASSES + 4 == RADIX_TOP_K_PASSES, "two score passes, four column passes");
  static_assert(RADIX_TOP_K_THREADS == RADIX_TOP_K_BUCKETS, "an invocation per bucket");
  if (subgroup_invocation() == 0) {
    radix_top_k_prefix = uint2(0, 0);
    radix_top_k_mask = uint2(0, 0);
    radix_top_k_rank = k - 1;
    radix_top_k_done = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  let sign_bits = 32 - 8 * RADIX_TOP_K_SCORE_PASSES;
  let bucket = 255 - subgroup_invocation();
  for (uint pass = 0; pass < passes; pass++) {
    if (radix_top_k_done != 0) { break; }
    let count = histograms[(row * RADIX_TOP_K_PASSES + pass) * RADIX_TOP_K_BUCKETS + bucket];
    let prefix = radix_top_k_prefix;
    let mask = radix_top_k_mask;
    let remaining = radix_top_k_rank;
    let shift = radix_top_k_shift(pass);
    uint keys;
    let higher = workgroup_prefix_sum<uint, RADIX_TOP_K_THREADS>(count, keys);
    if (higher <= remaining && remaining < higher + count) {
      var next_prefix = prefix;
      var next_mask = mask;
      if (pass < RADIX_TOP_K_SCORE_PASSES) {
        next_prefix.x |= bucket << shift;
        next_mask.x |= 0xffu << shift;
        if (pass == RADIX_TOP_K_SCORE_PASSES - 1) {
          next_prefix.x |= (next_prefix.x & 0x80000000u) != 0 ? 0u : (1u << sign_bits) - 1u;
          next_mask.x = 0xffffffffu;
        }
      } else {
        next_prefix.y |= bucket << shift;
        next_mask.y |= 0xffu << shift;
      }
      radix_top_k_prefix = next_prefix;
      radix_top_k_mask = next_mask;
      radix_top_k_rank = remaining - higher;
      radix_top_k_done = count == remaining - higher + 1 ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
  }
}

// This partition's columns of the row: an equal share of `columns`.
uint2 radix_top_k_partition(uint columns, uint partitions, uint partition) {
  return uint2(columns * partition / partitions, columns * (partition + 1) / partitions);
}

[[Kernel]]
[[Precondition("new", "columns > 0 && partitions > 0")]]
[[Precondition("encode", "rows > 0 && k > 0 && k <= 512 && k <= columns")]]
func RadixTopKSmallPass(
  Ptr<bf16, Access.Read> input,
  Ptr<uint, Access.ReadWrite> histograms,
  uint rows,
  uint k,
  uint pass,
  [[Specialize]] uint columns,
  [[Specialize]] uint partitions,
  [[Groups("partitions")]] uint partition,
  [[Groups("rows")]] uint row,
  [[Threads("256")]] uint tid,
) {
  radix_top_k_replay(histograms, row, k, pass);
  if (radix_top_k_done != 0) { return; }
  radix_top_k_histogram[tid] = 0;
  GroupMemoryBarrierWithGroupSync();
  let prefix = radix_top_k_prefix;
  let mask = radix_top_k_mask;
  let shift = radix_top_k_shift(pass);
  let score_pass = pass < RADIX_TOP_K_SCORE_PASSES;
  let range = radix_top_k_partition(columns, partitions, partition);
  for (uint column = range.x + tid; column < range.y; column += RADIX_TOP_K_THREADS) {
    let key = radix_top_k_key(input, row * columns, column);
    if (all((key & mask) == prefix)) {
      InterlockedAdd(radix_top_k_histogram[((score_pass ? key.x : key.y) >> shift) & 0xffu], 1u);
    }
  }
  GroupMemoryBarrierWithGroupSync();
  let count = radix_top_k_histogram[tid];
  if (count != 0) {
    InterlockedAdd(histograms[(row * RADIX_TOP_K_PASSES + pass) * RADIX_TOP_K_BUCKETS + tid], count);
  }
}

// Appends the row's k keys at or above the final prefix, in any order, each as its x and y words.
[[Kernel]]
[[Precondition("new", "columns > 0 && partitions > 0")]]
[[Precondition("encode", "rows > 0 && k > 0 && k <= 512 && k <= columns")]]
func RadixTopKSmallCollect(
  Ptr<bf16, Access.Read> input,
  Ptr<uint, Access.ReadWrite> histograms,
  Ptr<uint, Access.ReadWrite> counts,
  Ptr<uint, Access.ReadWrite> keys,
  uint rows,
  uint k,
  [[Specialize]] uint columns,
  [[Specialize]] uint partitions,
  [[Groups("partitions")]] uint partition,
  [[Groups("rows")]] uint row,
  [[Threads("256")]] uint tid,
) {
  radix_top_k_replay(histograms, row, k, RADIX_TOP_K_PASSES);
  let threshold = radix_top_k_prefix;
  let range = radix_top_k_partition(columns, partitions, partition);
  for (uint column = range.x + tid; column < range.y; column += RADIX_TOP_K_THREADS) {
    let key = radix_top_k_key(input, row * columns, column);
    if (!radix_top_k_above(threshold, key)) {
      uint slot;
      InterlockedAdd(counts[row], 1u, slot);
      // The final prefix admits exactly k unique keys.
      if (slot < k) {
        let at = 2 * (row * k + slot);
        keys[at] = key.x;
        keys[at + 1] = key.y;
      }
    }
  }
}

// Bitonic-sorts each row's k keys descending in shared memory and writes their columns and scores.
[[Kernel]]
[[Precondition("encode", "rows > 0 && k > 0 && k <= 512")]]
func RadixTopKSmallSort(
  Ptr<uint, Access.Read> keys,
  Ptr<uint, Access.ReadWrite> output_ids,
  Ptr<float, Access.ReadWrite> output_scores,
  uint rows,
  uint k,
  [[Groups("rows")]] uint row,
  [[Threads("256")]] uint tid,
) {
  for (uint index = tid; index < 512; index += RADIX_TOP_K_THREADS) {
    let at = 2 * (row * k + index);
    radix_top_k_keys[index] = index < k ? uint2(keys[at], keys[at + 1]) : uint2(0, 0);
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

  for (uint index = tid; index < k; index += RADIX_TOP_K_THREADS) {
    let key = radix_top_k_keys[index];
    output_ids[row * k + index] = ~key.y;
    output_scores[row * k + index] = asfloat(float_order_bits(key.x));
  }
}
