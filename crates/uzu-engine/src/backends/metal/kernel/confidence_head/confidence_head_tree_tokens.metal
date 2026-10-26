#include <metal_stdlib>
#include "../common/dsl.h"
#include "../weaver/weaver_frontier.h"
#include "../common/defines.h"
#include "../common/thread_context.h"
#include "confidence_head_common.h"

using namespace metal;

#define SLOTS_MAX 512u

// Map an expanded tree slot to its attention position, not its vocabulary token ID.
// The root occupies position 0. Other slots reuse their entry under their parent:
// position = 1 + parent_slot * children + child_index. Match by vocabulary ID within
// that parent's distinct children; CH_NONE means no matching saved child was found.
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

// Build attention inputs and topology, with 256 threads per attention position.
// Position 0 is the root; position 1 + entry represents a saved child, including children already expanded into slots.
// Unused positions are masked out and receive zero vectors.
// Each threadgroup independently gathers slot metadata for its position's topology lookup.
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
    threadgroup uint slot_token_of[SLOTS_MAX],
    threadgroup uint slot_parent[SLOTS_MAX],
    threadgroup uint slot_depth[SLOTS_MAX],
    const uint token GROUPS(1u + slot_count * children),
    const uint lid THREADS(HEAD_THREADS)
) {
  // 1. Cooperatively cache each valid slot's attention position, parent slot and depth.
  // Invalid slots use sentinels. metadata[3] will hold this position's expanded slot.
  const uint slots = slot_count, entries = slot_count * children;
  for (uint slot = lid; slot < slots; slot += HEAD_THREADS) {
    const bool valid = packed_tree[uint(TreeIdx::Valid) * slots + slot] != 0u;
    slot_token_of[slot] =
        valid ? confidence_head_slot_token(packed_tree, frontier, child_real, slot, slots, children) : CH_NONE;
    slot_parent[slot] = valid && slot != 0u ? packed_tree[uint(TreeIdx::ParentSlot) * slots + slot] : CH_NONE;
    slot_depth[slot] = valid ? packed_tree[uint(TreeIdx::Depth) * slots + slot] : 0u;
  }
  if (lid == 0) {
    metadata[3] = CH_NONE;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Find whether this attention position was expanded into a tree slot.
  // A valid tree has at most one matching slot; an unexpanded child keeps CH_NONE.
  // The preceding barrier publishes the cache; the next publishes the lookup result.
  for (uint slot = lid; slot < slots; slot += HEAD_THREADS) {
    if (slot_token_of[slot] == token && packed_tree[uint(TreeIdx::Valid) * slots + slot] != 0u) {
      metadata[3] = slot;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid == 0) {
    // 3. Resolve presence, depth and parent for this position.
    // Only thread 0 handles topology. Non-root positions use child_real, the root uses its tree-valid flag.
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
    // Saved node features are usable only if this slot has children recorded by encode_round.
    bool slot_has_children = false;
    if (slot != CH_NONE) {
      for (uint child = 0; child < children; ++child) {
        slot_has_children = slot_has_children || child_real[slot * children + child] != 0u;
      }
    }

    // 4. Build the ancestor table, indexed by absolute depth, including this position itself.
    // Follow parent slots to the root and store their attention positions.
    // Depth indices are clamped to max_depth, as are depth-embedding lookups below.
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

    // 5. Export topology and both mappings:
    // attention position -> expanded slot and expanded slot -> attention position.
    // Parent IDs are attention positions too.
    token_parents[token] = parent_slot == CH_NONE ? CH_NONE : slot_token_of[parent_slot];
    token_depths[token] = depth;
    token_present[token] = present ? 1u : 0u;
    token_slots[token] = slot;
    if (present && slot != CH_NONE) {
      slot_tokens[slot] = token;
    }
    // Broadcast presence, depth and the slot with saved node features to the other threads.
    metadata[0] = present ? 1u : 0u;
    metadata[1] = depth;
    metadata[2] = slot_has_children ? slot : CH_NONE;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 6. Assemble the width-dimensional attention input cooperatively after the barrier:
  // depth embedding + child projection (except root) + node projection (if saved).
  // An expanded position combines its child and parent roles in a single vector.
  // Missing positions receive zeros and are excluded as attention keys by token_present.
  const bool present = metadata[0] != 0u;
  const uint depth = metadata[1];
  const uint node_slot = metadata[2];
  for (uint unit = lid; unit < width; unit += HEAD_THREADS) {
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
