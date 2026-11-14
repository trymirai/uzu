#include <metal_stdlib>
#include "common/defines.h"
#include "common/dsl.h"
#include "generated/trie.h"
#include "generated/weaver.h"

using namespace metal;
using namespace uzu::trie;
using namespace uzu::weaver;

METAL_FUNC uint64_t prng_derive(uint64_t seed, uint64_t index) {
  uint64_t hash = seed + index;
  hash ^= hash >> 33;
  hash *= 0xff51afd7ed558ccdUL;
  hash ^= hash >> 33;
  hash *= 0xc4ceb9fe1a85ec53UL;
  hash ^= hash >> 33;
  return hash;
}

struct LinearizeOutputs {
  device TrieNode* nodes;
  device uint* token_ids;
  device uint64_t* seeds;
};

METAL_FUNC void linearize_emit(
    LinearizeOutputs out,
    bool with_seeds,
    uint flat,
    int parent_flat,
    uint token,
    uint depth,
    uint64_t prng_seed,
    uint root_position
) {
  out.nodes[flat] = TrieNode{flat, flat, depth, parent_flat};
  out.token_ids[flat] = token;
  if (with_seeds) {
    out.seeds[flat] = prng_derive(prng_seed, (uint64_t)root_position + depth);
  }
}

METAL_FUNC void linearize_pad(LinearizeOutputs out, bool with_seeds, uint flat) {
  out.nodes[flat] = TrieNode{0xffffffffU, 0, 0, -1};
  out.token_ids[flat] = 0;
  if (with_seeds) {
    out.seeds[flat] = 0;
  }
}

PUBLIC KERNEL(LinearizeChain)(
    const device uint* sampled,
    const device uint* target_output_token,
    device TrieNode* nodes,
    device uint* token_ids,
    device uint64_t* seeds OPTIONAL(with_seeds),
    constant const uint& chain_length,
    constant const uint& budget,
    constant const uint64_t& prng_seed,
    constant const uint& root_position,
    const bool with_seeds SPECIALIZE
) {
  LinearizeOutputs out{nodes, token_ids, seeds};

  const uint chain = min(chain_length + 1, budget);
  for (uint i = 0; i < budget; i++) {
    if (i < chain) {
      const uint token = (i == 0) ? target_output_token[0] : sampled[i - 1];
      linearize_emit(out, with_seeds, i, (i == 0) ? -1 : (int)(i - 1), token, i, prng_seed, root_position);
      out.nodes[i].trie_end = chain - 1;
    } else {
      linearize_pad(out, with_seeds, i);
    }
  }
}

#define LINEARIZE_SIMDGROUP_COUNT 8
#define LINEARIZE_THREADGROUP_SIZE (LINEARIZE_SIMDGROUP_COUNT * METAL_SIMD_SIZE)

// expand_width >= 1 bounds tree slots by the same limit as frontier slots.
UZU_CONST uint LINEARIZE_MAX_TREE_SLOTS = FRONTIER_MAX_SLOTS;
UZU_CONST uint LINEARIZE_MAX_SLOTS = LINEARIZE_MAX_TREE_SLOTS + FRONTIER_MAX_SLOTS;
UZU_CONST uint LINEARIZE_SLOTS_PER_THREAD =
    (LINEARIZE_MAX_SLOTS + LINEARIZE_THREADGROUP_SIZE - 1) / LINEARIZE_THREADGROUP_SIZE;

static_assert(LINEARIZE_SIMDGROUP_COUNT <= METAL_SIMD_SIZE, "winner reduction must fit in one SIMD group");
static_assert(LINEARIZE_SLOTS_PER_THREAD <= numeric_limits<uint>::digits, "selection masks must fit in uint");

METAL_FUNC uint linearize_winner(
    float logprob,
    uint slot,
    uint first_open,
    uint first_nan,
    threadgroup uint4* reduction,
    threadgroup uint& selected_slot,
    uint tid
) {
  constexpr uint NONE = FRONTIER_NO_WINNER;
  const uint lane = tid % METAL_SIMD_SIZE;
  const uint group = tid / METAL_SIMD_SIZE;
  const float maximum = simd_max(logprob);
  const uint winner = simd_min(logprob == maximum ? slot : NONE);
  first_open = simd_min(first_open);
  first_nan = simd_min(first_nan);
  if (lane == 0) {
    reduction[group] = uint4(as_type<uint>(maximum), winner, first_open, first_nan);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    const uint4 value =
        lane < LINEARIZE_SIMDGROUP_COUNT ? reduction[lane] : uint4(as_type<uint>(-INFINITY), NONE, NONE, NONE);
    const float group_logprob = as_type<float>(value.x);
    const float best_logprob = simd_max(group_logprob);
    const uint best = simd_min(group_logprob == best_logprob ? value.y : NONE);
    const uint first = simd_min(value.z);
    const uint nan = simd_min(value.w);
    if (lane == 0) {
      // A serial scan keeps its first open candidate when that score is NaN.
      selected_slot = first != NONE && first == nan ? first : best;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return selected_slot;
}

METAL_FUNC uint2
linearize_range(uint slot, uint kept, threadgroup const ushort* selected, threadgroup const ushort* parents) {
  uint before = 0;
  uint descendants = 0;
  for (uint index = 0; index < kept; index++) {
    const uint other = selected[index];
    uint2 ancestors(slot, other);
    uint2 children = ancestors;
    // Weaver appends children after parents, so the larger slot can walk up.
    while (ancestors.x != ancestors.y) {
      const uint side = uint(ancestors.x < ancestors.y);
      children[side] = ancestors[side];
      ancestors[side] = parents[ancestors[side]];
    }
    const uint common = ancestors.x;
    descendants += common == slot;
    before += common != slot && (common == other || children.y < children.x);
  }
  return uint2(before, before + descendants - 1);
}

PUBLIC KERNEL(LinearizeTree)(
    const device uint* packed_tree,
    const device uint* frontier,
    device TrieNode* nodes,
    device uint* token_ids,
    device uint64_t* seeds OPTIONAL(with_seeds),
    device uint* stack,
    constant const uint& tree_slot_count,
    constant const uint& frontier_capacity,
    constant const uint& budget,
    constant const uint64_t& prng_seed,
    constant const uint& root_position,
    const bool with_seeds SPECIALIZE,
    threadgroup ushort parent_slots[LINEARIZE_MAX_SLOTS],
    threadgroup ushort selected_slots[LINEARIZE_MAX_SLOTS],
    threadgroup uint4 reduction[LINEARIZE_SIMDGROUP_COUNT],
    threadgroup uint& selected_slot,
    const uint tid THREADS(LINEARIZE_THREADGROUP_SIZE)
) {
  constexpr uint NONE = FRONTIER_NO_WINNER;
  const uint total_slots = tree_slot_count + frontier_capacity;
  auto slot_field = [&](uint slot, TreeIdx tree_field, FrontierIdx frontier_field) -> uint {
    const bool tree = slot < tree_slot_count;
    const device uint* source = tree ? packed_tree : frontier;
    const uint stride = tree ? tree_slot_count : frontier_capacity;
    const uint index = tree ? slot : slot - tree_slot_count;
    const uint column = tree ? uint(tree_field) : uint(frontier_field);
    return source[column * stride + index];
  };

  uint parents[LINEARIZE_SLOTS_PER_THREAD];
  float path_logprobs[LINEARIZE_SLOTS_PER_THREAD];
  uint open_mask = 0;
  uint kept_mask = uint(tid == 0);
  METAL_PRAGMA_UNROLL
  for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
    parents[item] = NONE;
    path_logprobs[item] = -INFINITY;
    const uint slot = tid + item * LINEARIZE_THREADGROUP_SIZE;
    if (slot >= total_slots) {
      continue;
    }
    const bool valid = slot_field(slot, TreeIdx::Valid, FrontierIdx::Active) != 0;
    parents[item] = valid ? slot_field(slot, TreeIdx::ParentSlot, FrontierIdx::ParentSlot) : NONE;
    path_logprobs[item] = as_type<float>(slot_field(slot, TreeIdx::PathLogprobBits, FrontierIdx::PathLogprobBits));
    parent_slots[slot] = ushort(parents[item]);
    open_mask |= uint(parents[item] == 0) << item;
  }
  if (tid == 0) {
    selected_slots[0] = 0;
  }

  uint kept = 1;
  while (kept < budget) {
    float best_logprob = -INFINITY;
    uint best = NONE;
    uint first_open = NONE;
    uint first_nan = NONE;
    METAL_PRAGMA_UNROLL
    for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
      if ((open_mask & (1u << item)) == 0) {
        continue;
      }
      const uint slot = tid + item * LINEARIZE_THREADGROUP_SIZE;
      const float logprob = path_logprobs[item];
      first_open = min(first_open, slot);
      first_nan = min(first_nan, isnan(logprob) ? slot : NONE);
      const bool better = logprob > best_logprob || (logprob == best_logprob && slot < best);
      best = better ? slot : best;
      best_logprob = better ? logprob : best_logprob;
    }
    best = linearize_winner(best_logprob, best, first_open, first_nan, reduction, selected_slot, tid);
    if (best == NONE) {
      break;
    }
    if (tid == 0) {
      selected_slots[kept] = ushort(best);
    }
    uint children = 0;
    METAL_PRAGMA_UNROLL
    for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
      const uint slot = tid + item * LINEARIZE_THREADGROUP_SIZE;
      kept_mask |= uint(slot == best) << item;
      children |= uint(parents[item] == best) << item;
    }
    open_mask = (open_mask | children) & ~kept_mask;
    kept++;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  LinearizeOutputs out{nodes, token_ids, seeds};
  METAL_PRAGMA_UNROLL
  for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
    if ((kept_mask & (1u << item)) == 0) {
      continue;
    }
    const uint slot = tid + item * LINEARIZE_THREADGROUP_SIZE;
    const uint2 range = linearize_range(slot, kept, selected_slots, parent_slots);
    const uint token = slot_field(slot, TreeIdx::TokenId, FrontierIdx::TokenId);
    const uint depth = slot_field(slot, TreeIdx::Depth, FrontierIdx::Depth);
    linearize_emit(out, with_seeds, range.x, -1, token, depth, prng_seed, root_position);
    nodes[range.x].trie_end = range.y;
    // Reuse the existing scratch buffer as the slot-to-flat map.
    stack[slot] = range.x;
  }
  threadgroup_barrier(mem_flags::mem_device);
  METAL_PRAGMA_UNROLL
  for (uint item = 0; item < LINEARIZE_SLOTS_PER_THREAD; item++) {
    if ((kept_mask & (1u << item)) == 0) {
      continue;
    }
    const uint slot = tid + item * LINEARIZE_THREADGROUP_SIZE;
    const uint parent = parents[item];
    nodes[stack[slot]].parent = parent == NONE ? -1 : int(stack[parent]);
  }
  for (uint flat = kept + tid; flat < budget; flat += LINEARIZE_THREADGROUP_SIZE) {
    linearize_pad(out, with_seeds, flat);
  }
}
