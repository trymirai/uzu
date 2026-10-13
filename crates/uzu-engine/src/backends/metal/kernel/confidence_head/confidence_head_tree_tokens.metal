#include <metal_stdlib>
#include "../common/dsl.h"
#include "../weaver/weaver_frontier.h"
#include "confidence_head.h"

using namespace metal;

#define CH_LEVELS_MAX 32u
#define CH_SLOTS_MAX 512u

// The token of a slot: the root's, or that of its entry under its parent, found by its token among the parent's
// children (pool tokens are distinct).
METAL_FUNC uint confidence_head_slot_token(
    const device uint* packed_tree,
    const device uint* frontier,
    const device uint* child_real,
    uint slot,
    uint slots,
    uint children
) {
  if (slot == 0u) {
    return 0u;
  }
  const uint parent = packed_tree[uint(TreeIdx::ParentSlot) * slots + slot];
  const uint token_id = packed_tree[uint(TreeIdx::TokenId) * slots + slot];
  for (uint child = 0; child < children; ++child) {
    const uint index = parent * children + child;
    if (child_real[index] != 0u && frontier[uint(FrontierIdx::TokenId) * slots * children + index] == token_id) {
      return 1u + index;
    }
  }
  return CH_NONE;
}

// The tokens of the head's tree attention, one threadgroup per token: the root, then one token per frontier entry.
// See the CPU twin for the contract. Every slot's token, parent and depth are first gathered in parallel into
// threadgroup memory, so the token's own slot is found and its ancestors walked without serial device reads.
PUBLIC KERNEL(ConfidenceHeadTreeTokens)(
    const device uint* packed_tree,
    const device uint* frontier,
    const device uint* child_real,
    const device float* node_inputs,
    const device float* child_inputs,
    const device float* depth_embedding,
    device float* tokens,
    device uint* token_parents,
    device uint* token_depths,
    device uint* token_present,
    device uint* token_slots,
    device uint* token_ancestors,
    device uint* slot_tokens,
    constant uint& slot_count,
    constant uint& children,
    constant uint& width,
    constant uint& max_depth,
    threadgroup uint metadata[4],
    threadgroup uint slot_token_of[CH_SLOTS_MAX],
    threadgroup uint slot_parent[CH_SLOTS_MAX],
    threadgroup uint slot_depth[CH_SLOTS_MAX],
    const uint token GROUPS(1u + slot_count * children),
    const uint lid THREADS(CH_THREADS)
) {
  const uint slots = slot_count, entries = slot_count * children;
  for (uint slot = lid; slot < slots; slot += CH_THREADS) {
    slot_token_of[slot] = confidence_head_slot_token(packed_tree, frontier, child_real, slot, slots, children);
    slot_parent[slot] = slot == 0u ? CH_NONE : packed_tree[uint(TreeIdx::ParentSlot) * slots + slot];
    slot_depth[slot] = packed_tree[uint(TreeIdx::Depth) * slots + slot];
  }
  if (lid == 0) {
    metadata[3] = CH_NONE;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The slot this token was expanded into: the valid slot whose token it is.
  for (uint slot = lid; slot < slots; slot += CH_THREADS) {
    if (slot_token_of[slot] == token && packed_tree[uint(TreeIdx::Valid) * slots + slot] != 0u) {
      metadata[3] = slot;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    bool present;
    uint depth = 0u, parent_slot = CH_NONE, slot = CH_NONE;
    if (token == 0u) {
      present = packed_tree[uint(TreeIdx::Valid) * slots] != 0u;
      slot = 0u;
    } else {
      const uint index = token - 1u;
      present = child_real[index] != 0u;
      if (present) {
        depth = frontier[uint(FrontierIdx::Depth) * entries + index];
        parent_slot = index / children;
        slot = metadata[3];
      }
    }
    bool slot_has_children = false;
    if (slot != CH_NONE) {
      for (uint child = 0; child < children; ++child) {
        slot_has_children = slot_has_children || child_real[slot * children + child] != 0u;
      }
    }
    const uint levels = max_depth + 1u;
    for (uint level = 0; level < levels; ++level) {
      token_ancestors[token * levels + level] = CH_NONE;
    }
    if (present) {
      token_ancestors[token * levels + min(depth, max_depth)] = token;
      uint walk = parent_slot;
      while (walk != CH_NONE) {
        token_ancestors[token * levels + min(slot_depth[walk], max_depth)] = slot_token_of[walk];
        walk = slot_parent[walk];
      }
    }
    token_parents[token] = parent_slot == CH_NONE ? CH_NONE : slot_token_of[parent_slot];
    token_depths[token] = depth;
    token_present[token] = present ? 1u : 0u;
    token_slots[token] = slot;
    if (present && slot != CH_NONE) {
      slot_tokens[slot] = token;
    }
    metadata[0] = present ? 1u : 0u;
    metadata[1] = depth;
    metadata[2] = slot_has_children ? slot : CH_NONE;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool present = metadata[0] != 0u;
  const uint depth = metadata[1], node_slot = metadata[2];
  for (uint unit = lid; unit < width; unit += CH_THREADS) {
    float value = 0.0f;
    if (present) {
      value = depth_embedding[min(depth, max_depth) * width + unit];
      if (token > 0u) {
        value += child_inputs[(token - 1u) * width + unit];
      }
      if (node_slot != CH_NONE) {
        value += node_inputs[node_slot * width + unit];
      }
    }
    tokens[token * width + unit] = value;
  }
}
