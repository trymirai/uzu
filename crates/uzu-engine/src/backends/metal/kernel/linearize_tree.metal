#include <metal_stdlib>
#include "common/defines.h"
#include "common/dsl.h"
#include "generated/trie.h"
#include "generated/weaver.h"

using namespace metal;
using namespace uzu::trie;
using namespace uzu::weaver;

#define THREADGROUP_SIZE 256

inline uint64_t prng_derive(uint64_t seed, uint64_t index) {
  uint64_t hash = seed + index;
  hash ^= hash >> 33;
  hash *= 0xff51afd7ed558ccdUL;
  hash ^= hash >> 33;
  hash *= 0xc4ceb9fe1a85ec53UL;
  hash ^= hash >> 33;
  return hash;
}

PUBLIC KERNEL(LinearizeTree)(
    const device uint* packed_tree,
    const device uint* frontier,
    const device uint* sampled,
    const device uint* target_output_token,
    device TrieNode* nodes,
    device int* parents,
    device uint* heights,
    device uint* token_ids,
    device uint64_t* seeds,
    device uint* count,
    device uint* stack,
    constant const uint& tree_slot_count,
    constant const uint& frontier_capacity,
    constant const uint& budget,
    constant const uint& chain_length,
    constant const uint& mode,
    constant const uint64_t& prng_seed,
    constant const uint& root_position,
    const uint thread_index THREADS(THREADGROUP_SIZE)
) {
  if (thread_index != 0) {
    return;
  }

  const uint tree_slots = tree_slot_count;
  const uint frontier_slots = frontier_capacity;
  const uint total_slots = tree_slots + frontier_slots;

  auto tree_field = [&](uint field, uint slot) -> uint { return packed_tree[field * tree_slots + slot]; };
  auto frontier_field = [&](uint field, uint slot) -> uint { return frontier[field * frontier_slots + slot]; };

  auto slot_valid = [&](uint slot) -> bool {
    if (slot < tree_slots) {
      return tree_field((uint)TreeIdx::Valid, slot) != 0;
    }
    return frontier_field((uint)FrontierIdx::Active, slot - tree_slots) != 0;
  };
  // Parent references are always tree-slot indices (FRONTIER_NO_WINNER for the root).
  auto slot_parent = [&](uint slot) -> uint {
    if (slot < tree_slots) {
      return tree_field((uint)TreeIdx::ParentSlot, slot);
    }
    return frontier_field((uint)FrontierIdx::ParentSlot, slot - tree_slots);
  };
  auto slot_token = [&](uint slot) -> uint {
    if (slot < tree_slots) {
      return tree_field((uint)TreeIdx::TokenId, slot);
    }
    return frontier_field((uint)FrontierIdx::TokenId, slot - tree_slots);
  };
  auto slot_depth = [&](uint slot) -> uint {
    if (slot < tree_slots) {
      return tree_field((uint)TreeIdx::Depth, slot);
    }
    return frontier_field((uint)FrontierIdx::Depth, slot - tree_slots);
  };
  auto slot_path_logprob = [&](uint slot) -> uint {
    if (slot < tree_slots) {
      return tree_field((uint)TreeIdx::PathLogprobBits, slot);
    }
    return frontier_field((uint)FrontierIdx::PathLogprobBits, slot - tree_slots);
  };

  auto emit = [&](uint flat, int parent_flat, uint token, uint depth) {
    nodes[flat] = TrieNode{flat, flat, depth};
    parents[flat] = parent_flat;
    heights[flat] = depth;
    token_ids[flat] = token;
    seeds[flat] = prng_derive(prng_seed, (uint64_t)root_position + depth);
  };
  auto pad = [&](uint flat) {
    nodes[flat] = TrieNode{0xffffffffU, 0, 0};
    parents[flat] = -1;
    heights[flat] = 0;
    token_ids[flat] = 0;
    seeds[flat] = 0;
  };

  if (mode == 1) {
    const uint chain = min(chain_length + 1, budget);
    for (uint i = 0; i < budget; i++) {
      if (i < chain) {
        const uint token = (i == 0) ? target_output_token[0] : sampled[i - 1];
        emit(i, (i == 0) ? -1 : (int)(i - 1), token, i);
        nodes[i].trie_end = chain - 1;
      } else {
        pad(i);
      }
    }
    *count = chain;
    return;
  }

  device uint* state = stack;
  device uint* dfs_stack = stack + total_slots;
  const uint OPEN = 1;
  const uint KEPT = 2;

  for (uint slot = 0; slot < total_slots; slot++) {
    state[slot] = 0;
  }
  state[0] = KEPT;
  uint kept = 1;
  for (uint child = 0; child < total_slots; child++) {
    if (slot_valid(child) && slot_parent(child) == 0 && slot_parent(child) != FRONTIER_NO_WINNER) {
      state[child] = OPEN;
    }
  }
  while (kept < budget) {
    int best = -1;
    float best_logprob = 0.0;
    for (uint slot = 0; slot < total_slots; slot++) {
      if (state[slot] != OPEN) {
        continue;
      }
      const float logprob = as_type<float>(slot_path_logprob(slot));
      if (best < 0 || logprob > best_logprob) {
        best = (int)slot;
        best_logprob = logprob;
      }
    }
    if (best < 0) {
      break;
    }
    state[best] = KEPT;
    kept++;
    for (uint child = 0; child < total_slots; child++) {
      if (slot_valid(child) && slot_parent(child) == (uint)best && slot_parent(child) != FRONTIER_NO_WINNER) {
        state[child] = OPEN;
      }
    }
  }

  emit(0, -1, slot_token(0), slot_depth(0));
  uint emitted = 1;

  uint stack_depth = 1;
  dfs_stack[0] = 0;
  dfs_stack[1] = 0;
  dfs_stack[2] = 0;

  while (stack_depth > 0) {
    const uint base = (stack_depth - 1) * 3;
    const uint parent_slot = dfs_stack[base];
    const uint parent_flat = dfs_stack[base + 1];
    const uint cursor = dfs_stack[base + 2];

    uint child = cursor;
    while (child < total_slots) {
      if (slot_valid(child) && slot_parent(child) == parent_slot && slot_parent(child) != FRONTIER_NO_WINNER &&
          state[child] == KEPT) {
        break;
      }
      child++;
    }

    if (child < total_slots) {
      dfs_stack[base + 2] = child + 1;
      const uint flat = emitted;
      emit(flat, (int)parent_flat, slot_token(child), slot_depth(child));
      emitted++;
      dfs_stack[stack_depth * 3] = child;
      dfs_stack[stack_depth * 3 + 1] = flat;
      dfs_stack[stack_depth * 3 + 2] = 0;
      stack_depth++;
    } else {
      nodes[parent_flat].trie_end = emitted - 1;
      stack_depth--;
    }
  }

  for (uint i = emitted; i < budget; i++) {
    pad(i);
  }
  *count = emitted;
}
