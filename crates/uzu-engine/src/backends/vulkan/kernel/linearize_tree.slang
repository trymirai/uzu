import definitions;
import common.ranking;
import common.workgroup_reduction;
import generated.trie;
import generated.weaver;

// Int64 is not a capability every device enables, so 64-bit seeds travel as (low, high) word pairs: the seed
// argument split in two by the host wrapper, and two words per entry of the `seeds` buffer.

uint2 linearize_add64(uint2 a, uint2 b) {
  let low = a.x + b.x;
  return uint2(low, a.y + b.y + (low < a.x ? 1u : 0u));
}

uint linearize_mul_high32(uint a, uint b) {
  let a_low = a & 0xffffu;
  let a_high = a >> 16;
  let b_low = b & 0xffffu;
  let b_high = b >> 16;
  let low = a_low * b_low;
  let cross_a = a_high * b_low;
  let cross_b = a_low * b_high;
  let carry = ((low >> 16) + (cross_a & 0xffffu) + (cross_b & 0xffffu)) >> 16;
  return a_high * b_high + (cross_a >> 16) + (cross_b >> 16) + carry;
}

uint2 linearize_mul64(uint2 a, uint2 b) {
  return uint2(a.x * b.x, linearize_mul_high32(a.x, b.x) + a.x * b.y + a.y * b.x);
}

uint2 linearize_shift_xor33(uint2 hash) {
  return uint2(hash.x ^ (hash.y >> 1), hash.y);
}

// Must stay bit-identical to `prng_derive` in the CPU and Metal kernels.
uint2 linearize_prng_derive(uint2 seed, uint2 index) {
  var hash = linearize_add64(seed, index);
  hash = linearize_shift_xor33(hash);
  hash = linearize_mul64(hash, uint2(0xed558ccdu, 0xff51afd7u));
  hash = linearize_shift_xor33(hash);
  hash = linearize_mul64(hash, uint2(0x1a85ec53u, 0xc4ceb9feu));
  return linearize_shift_xor33(hash);
}

void linearize_emit(
  Ptr<TrieNode, Access.ReadWrite> nodes,
  Ptr<uint, Access.ReadWrite> token_ids,
  Ptr<uint, Access.ReadWrite> seeds,
  bool with_seeds,
  uint flat,
  uint trie_end,
  int parent_flat,
  uint token,
  uint depth,
  uint2 prng_seed,
  uint root_position,
) {
  TrieNode node;
  node.trie_start = flat;
  node.trie_end = trie_end;
  node.height = depth;
  node.parent = parent_flat;
  nodes[flat] = node;
  token_ids[flat] = token;
  if (with_seeds) {
    let seed = linearize_prng_derive(prng_seed, linearize_add64(uint2(root_position, 0), uint2(depth, 0)));
    seeds[flat * 2] = seed.x;
    seeds[flat * 2 + 1] = seed.y;
  }
}

void linearize_pad(
  Ptr<TrieNode, Access.ReadWrite> nodes,
  Ptr<uint, Access.ReadWrite> token_ids,
  Ptr<uint, Access.ReadWrite> seeds,
  bool with_seeds,
  uint flat,
) {
  TrieNode node;
  node.trie_start = 0xffffffffu;
  node.trie_end = 0;
  node.height = 0;
  node.parent = -1;
  nodes[flat] = node;
  token_ids[flat] = 0;
  if (with_seeds) {
    seeds[flat * 2] = 0;
    seeds[flat * 2 + 1] = 0;
  }
}

[[Kernel]]
func LinearizeChain(
  Ptr<uint, Access.Read> sampled,
  Ptr<uint, Access.Read> target_output_token,
  Ptr<TrieNode, Access.ReadWrite> nodes,
  Ptr<uint, Access.ReadWrite> token_ids,
  [[Optional("with_seeds")]] Ptr<uint, Access.ReadWrite> seeds,
  uint chain_length,
  uint budget,
  uint prng_seed_low,
  uint prng_seed_high,
  uint root_position,
  [[Axis("budget", "64")]] uint flat,
  [[Specialize]] bool with_seeds,
) {
  let chain = min(chain_length + 1, budget);
  if (flat >= chain) {
    linearize_pad(nodes, token_ids, seeds, with_seeds, flat);
    return;
  }
  let token = flat == 0 ? target_output_token[0] : sampled[flat - 1];
  let parent = flat == 0 ? -1 : int(flat - 1);
  linearize_emit(
    nodes, token_ids, seeds, with_seeds, flat, chain - 1, parent, token, flat, uint2(prng_seed_low, prng_seed_high),
    root_position
  );
}

static const uint LINEARIZE_THREADS = 256;
// expand_width >= 1 bounds tree slots by the same limit as frontier slots.
static const uint LINEARIZE_MAX_SLOTS = 2 * FRONTIER_MAX_SLOTS;
static const uint LINEARIZE_SLOTS_PER_THREAD = (LINEARIZE_MAX_SLOTS + LINEARIZE_THREADS - 1) / LINEARIZE_THREADS;

// 16-bit slots keep both arrays within the 16 KiB of shared memory every device provides.
groupshared uint16_t linearize_parents[LINEARIZE_MAX_SLOTS];
// The selection order while selecting; afterwards, each kept slot's flat index.
groupshared uint16_t linearize_slots[LINEARIZE_MAX_SLOTS];

// (score order key, best slot, first open slot, first open slot with a NaN score)
struct LinearizeWinner : IReduction {
  typealias Value = uint4;
  static uint4 combine(uint4 a, uint4 b) {
    let best = b.x > a.x || (b.x == a.x && b.y < a.y) ? b.xy : a.xy;
    return uint4(best, min(a.z, b.z), min(a.w, b.w));
  }
  static uint4 subgroup(uint4 value) {
    let key = WaveActiveMax(value.x);
    let best = WaveActiveMin(value.x == key ? value.y : FRONTIER_NO_WINNER);
    return uint4(key, best, WaveActiveMin(value.z), WaveActiveMin(value.w));
  }
}

// The flat range [first, last] of `slot`'s subtree in the depth-first order of the kept slots.
uint2 linearize_range(uint slot, uint kept) {
  uint before = 0;
  uint descendants = 0;
  for (uint index = 0; index < kept; index++) {
    let other = uint(linearize_slots[index]);
    var own_ancestor = slot;
    var other_ancestor = other;
    var own_child = slot;
    var other_child = other;
    // Weaver appends children after parents, so the larger slot can walk up.
    while (own_ancestor != other_ancestor) {
      if (own_ancestor < other_ancestor) {
        other_child = other_ancestor;
        other_ancestor = uint(linearize_parents[other_ancestor]);
      } else {
        own_child = own_ancestor;
        own_ancestor = uint(linearize_parents[own_ancestor]);
      }
    }
    let common = own_ancestor;
    descendants += common == slot ? 1u : 0u;
    before += common != slot && (common == other || other_child < own_child) ? 1u : 0u;
  }
  return uint2(before, before + descendants - 1);
}

[[Kernel]]
func LinearizeTree(
  Ptr<uint, Access.Read> packed_tree,
  Ptr<uint, Access.Read> frontier,
  Ptr<TrieNode, Access.ReadWrite> nodes,
  Ptr<uint, Access.ReadWrite> token_ids,
  [[Optional("with_seeds")]] Ptr<uint, Access.ReadWrite> seeds,
  uint tree_slot_count,
  uint frontier_capacity,
  uint budget,
  uint prng_seed_low,
  uint prng_seed_high,
  uint root_position,
  // One workgroup selects the whole tree.
  [[Groups("1")]] uint group,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool with_seeds,
) {
  static_assert(LINEARIZE_SLOTS_PER_THREAD <= 32, "selection masks must fit in uint");
  let total_slots = tree_slot_count + frontier_capacity;

  let root_score = asfloat(packed_tree[uint(TreeIdx.EdgeLogprobBits) * tree_slot_count]);
  uint parents[LINEARIZE_SLOTS_PER_THREAD];
  float scores[LINEARIZE_SLOTS_PER_THREAD];
  uint open_mask = 0;
  uint kept_mask = tid == 0 ? 1u : 0u;
  for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
    parents[item] = FRONTIER_NO_WINNER;
    scores[item] = NEGATIVE_INFINITY;
    let slot = tid + item * LINEARIZE_THREADS;
    if (slot >= total_slots) {
      continue;
    }
    let tree = slot < tree_slot_count;
    let source = tree ? packed_tree : frontier;
    let stride = tree ? tree_slot_count : frontier_capacity;
    let index = tree ? slot : slot - tree_slot_count;
    let valid = source[(tree ? uint(TreeIdx.Valid) : uint(FrontierIdx.Active)) * stride + index] != 0;
    if (valid) {
      parents[item] = source[(tree ? uint(TreeIdx.ParentSlot) : uint(FrontierIdx.ParentSlot)) * stride + index];
    }
    let edge =
      asfloat(source[(tree ? uint(TreeIdx.EdgeLogprobBits) : uint(FrontierIdx.EdgeLogprobBits)) * stride + index]);
    scores[item] = parents[item] == 0 ? root_score + edge : edge;
    linearize_parents[slot] = uint16_t(parents[item]);
    open_mask |= (parents[item] == 0 ? 1u : 0u) << item;
  }
  if (tid == 0) {
    linearize_slots[0] = 0;
  }

  uint kept = 1;
  while (kept < budget) {
    var best_logprob = NEGATIVE_INFINITY;
    var best = FRONTIER_NO_WINNER;
    var first_open = FRONTIER_NO_WINNER;
    var first_nan = FRONTIER_NO_WINNER;
    for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
      if ((open_mask & (1u << item)) == 0) {
        continue;
      }
      let slot = tid + item * LINEARIZE_THREADS;
      let logprob = scores[item];
      first_open = min(first_open, slot);
      first_nan = min(first_nan, isnan(logprob) ? slot : FRONTIER_NO_WINNER);
      let better = logprob > best_logprob || (logprob == best_logprob && slot < best);
      best = better ? slot : best;
      best_logprob = better ? logprob : best_logprob;
    }
    let winner = workgroup_reduce<LinearizeWinner, LINEARIZE_THREADS>(
      uint4(float_order_key(asuint(best_logprob)), best, first_open, first_nan)
    );
    // A serial scan keeps its first open candidate when that score is NaN.
    let keep_nan = winner.z != FRONTIER_NO_WINNER && winner.z == winner.w;
    best = keep_nan ? winner.z : winner.y;
    if (best == FRONTIER_NO_WINNER) {
      break;
    }
    let best_score = keep_nan ? asfloat(0x7fc00000u) : asfloat(float_order_bits(winner.x));
    if (tid == 0) {
      linearize_slots[kept] = uint16_t(best);
    }
    uint children = 0;
    for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
      let slot = tid + item * LINEARIZE_THREADS;
      kept_mask |= (slot == best ? 1u : 0u) << item;
      let child = parents[item] == best;
      children |= (child ? 1u : 0u) << item;
      scores[item] = child ? best_score + scores[item] : scores[item];
    }
    open_mask = (open_mask | children) & ~kept_mask;
    kept++;
  }
  GroupMemoryBarrierWithGroupSync();

  uint2 ranges[LINEARIZE_SLOTS_PER_THREAD];
  for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
    if ((kept_mask & (1u << item)) != 0) {
      ranges[item] = linearize_range(tid + item * LINEARIZE_THREADS, kept);
    }
  }
  // Every invocation has finished reading the selection order before it is overwritten with flat indices.
  GroupMemoryBarrierWithGroupSync();
  for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
    if ((kept_mask & (1u << item)) != 0) {
      linearize_slots[tid + item * LINEARIZE_THREADS] = uint16_t(ranges[item].x);
    }
  }
  GroupMemoryBarrierWithGroupSync();

  let prng_seed = uint2(prng_seed_low, prng_seed_high);
  for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
    if ((kept_mask & (1u << item)) == 0) {
      continue;
    }
    let slot = tid + item * LINEARIZE_THREADS;
    let tree = slot < tree_slot_count;
    let source = tree ? packed_tree : frontier;
    let stride = tree ? tree_slot_count : frontier_capacity;
    let index = tree ? slot : slot - tree_slot_count;
    let token = source[(tree ? uint(TreeIdx.TokenId) : uint(FrontierIdx.TokenId)) * stride + index];
    let depth = source[(tree ? uint(TreeIdx.Depth) : uint(FrontierIdx.Depth)) * stride + index];
    // Kept slots only ever open their own children, so a kept slot's parent is kept too.
    let parent = parents[item];
    let parent_flat = parent == FRONTIER_NO_WINNER ? -1 : int(linearize_slots[parent]);
    linearize_emit(
      nodes, token_ids, seeds, with_seeds, ranges[item].x, ranges[item].y, parent_flat, token, depth, prng_seed,
      root_position
    );
  }
  for (uint flat = kept + tid; flat < budget; flat += LINEARIZE_THREADS) {
    linearize_pad(nodes, token_ids, seeds, with_seeds, flat);
  }
}
