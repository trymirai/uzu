use uzu_engine_macros::kernel;

use crate::backends::common::gpu_types::{
    trie::TrieNode,
    weaver::{FRONTIER_NO_WINNER, FrontierIdx, TreeIdx},
};

fn prng_derive(
    seed: u64,
    index: u64,
) -> u64 {
    let mut hash = seed.wrapping_add(index);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;
    hash
}

#[allow(clippy::too_many_arguments)]
unsafe fn emit_node(
    nodes: *mut TrieNode,
    token_ids: *mut u32,
    seeds: *mut u64,
    flat: usize,
    parent_flat: i32,
    token: u32,
    depth: u32,
    prng_seed: u64,
    root_position: u32,
) {
    unsafe {
        *nodes.add(flat) = TrieNode {
            trie_start: flat as u32,
            trie_end: flat as u32,
            height: depth,
            parent: parent_flat,
        };
        *token_ids.add(flat) = token;
        *seeds.add(flat) = prng_derive(prng_seed, root_position as u64 + depth as u64);
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn pad_node(
    nodes: *mut TrieNode,
    token_ids: *mut u32,
    seeds: *mut u64,
    flat: usize,
) {
    unsafe {
        *nodes.add(flat) = TrieNode {
            trie_start: u32::MAX,
            trie_end: 0,
            height: 0,
            parent: -1,
        };
        *token_ids.add(flat) = 0;
        *seeds.add(flat) = 0;
    }
}

#[kernel(LinearizeChain)]
pub fn linearize_chain(
    sampled: *const u32,
    target_output_token: *const u32,
    nodes: *mut TrieNode,
    token_ids: *mut u32,
    seeds: *mut u64,
    count: *mut u32,
    chain_length: u32,
    budget: u32,
    prng_seed: u64,
    root_position: u32,
) {
    let budget = budget as usize;
    let chain = (chain_length as usize + 1).min(budget);
    for i in 0..budget {
        if i < chain {
            let token = unsafe {
                if i == 0 {
                    *target_output_token
                } else {
                    *sampled.add(i - 1)
                }
            };
            unsafe {
                emit_node(
                    nodes,
                    token_ids,
                    seeds,
                    i,
                    if i == 0 {
                        -1
                    } else {
                        (i - 1) as i32
                    },
                    token,
                    i as u32,
                    prng_seed,
                    root_position,
                );
                (*nodes.add(i)).trie_end = (chain - 1) as u32;
            }
        } else {
            unsafe { pad_node(nodes, token_ids, seeds, i) };
        }
    }
    unsafe {
        *count = chain as u32;
    }
}

#[kernel(LinearizeTree)]
pub fn linearize_tree(
    packed_tree: *const u32,
    frontier: *const u32,
    nodes: *mut TrieNode,
    token_ids: *mut u32,
    seeds: *mut u64,
    count: *mut u32,
    stack: *mut u32,
    tree_slot_count: u32,
    frontier_capacity: u32,
    budget: u32,
    prng_seed: u64,
    root_position: u32,
) {
    let tree_slots = tree_slot_count as usize;
    let frontier_slots = frontier_capacity as usize;
    let total_slots = tree_slots + frontier_slots;
    let budget = budget as usize;

    let tree_field = |field: usize, slot: usize| -> u32 {
        debug_assert!(slot < tree_slots);
        unsafe { *packed_tree.add(field * tree_slots + slot) }
    };
    let frontier_field = |field: usize, slot: usize| -> u32 {
        debug_assert!(slot < frontier_slots);
        unsafe { *frontier.add(field * frontier_slots + slot) }
    };

    let slot_valid = |slot: usize| -> bool {
        if slot < tree_slots {
            tree_field(TreeIdx::Valid as usize, slot) != 0
        } else {
            frontier_field(FrontierIdx::Active as usize, slot - tree_slots) != 0
        }
    };
    // Parent references are always tree-slot indices (FRONTIER_NO_WINNER for the root).
    let slot_parent = |slot: usize| -> u32 {
        if slot < tree_slots {
            tree_field(TreeIdx::ParentSlot as usize, slot)
        } else {
            frontier_field(FrontierIdx::ParentSlot as usize, slot - tree_slots)
        }
    };
    let slot_token = |slot: usize| -> u32 {
        if slot < tree_slots {
            tree_field(TreeIdx::TokenId as usize, slot)
        } else {
            frontier_field(FrontierIdx::TokenId as usize, slot - tree_slots)
        }
    };
    let slot_depth = |slot: usize| -> u32 {
        if slot < tree_slots {
            tree_field(TreeIdx::Depth as usize, slot)
        } else {
            frontier_field(FrontierIdx::Depth as usize, slot - tree_slots)
        }
    };
    let slot_path_logprob = |slot: usize| -> u32 {
        if slot < tree_slots {
            tree_field(TreeIdx::PathLogprobBits as usize, slot)
        } else {
            frontier_field(FrontierIdx::PathLogprobBits as usize, slot - tree_slots)
        }
    };

    debug_assert!(slot_valid(0), "tree slot 0 must be the valid root");

    const OPEN: u32 = 1;
    const KEPT: u32 = 2;

    // `stack` holds one selection-state word per slot, then the DFS emission stack.
    let state = stack;
    let dfs_stack = unsafe { stack.add(total_slots) };

    for slot in 0..total_slots {
        unsafe { *state.add(slot) = 0 };
    }
    unsafe { *state.add(0) = KEPT };
    let mut kept = 1usize;
    for child in 0..total_slots {
        if slot_valid(child) && slot_parent(child) == 0 && slot_parent(child) != FRONTIER_NO_WINNER {
            unsafe { *state.add(child) = OPEN };
        }
    }
    while kept < budget {
        let mut best: Option<usize> = None;
        let mut best_logprob = 0.0f32;
        for slot in 0..total_slots {
            if unsafe { *state.add(slot) } != OPEN {
                continue;
            }
            let logprob = f32::from_bits(slot_path_logprob(slot));
            if best.is_none() || logprob > best_logprob {
                best = Some(slot);
                best_logprob = logprob;
            }
        }
        let Some(best) = best else {
            break;
        };
        unsafe { *state.add(best) = KEPT };
        kept += 1;
        for child in 0..total_slots {
            if slot_valid(child) && slot_parent(child) == best as u32 && slot_parent(child) != FRONTIER_NO_WINNER {
                unsafe { *state.add(child) = OPEN };
            }
        }
    }

    unsafe {
        emit_node(nodes, token_ids, seeds, 0, -1, slot_token(0), slot_depth(0), prng_seed, root_position);
    }
    let mut emitted = 1usize;

    unsafe {
        *dfs_stack.add(0) = 0; // root slot
        *dfs_stack.add(1) = 0; // root flat index
        *dfs_stack.add(2) = 0; // scan cursor
    }
    let mut stack_depth = 1usize;

    while stack_depth > 0 {
        let base = (stack_depth - 1) * 3;
        let parent_slot = unsafe { *dfs_stack.add(base) } as usize;
        let parent_flat = unsafe { *dfs_stack.add(base + 1) };
        let cursor = unsafe { *dfs_stack.add(base + 2) as usize };

        let mut child = cursor;
        while child < total_slots {
            if slot_valid(child)
                && slot_parent(child) == parent_slot as u32
                && slot_parent(child) != FRONTIER_NO_WINNER
                && unsafe { *state.add(child) } == KEPT
            {
                break;
            }
            child += 1;
        }

        if child < total_slots {
            unsafe {
                *dfs_stack.add(base + 2) = (child + 1) as u32;
            }
            let flat = emitted;
            unsafe {
                emit_node(
                    nodes,
                    token_ids,
                    seeds,
                    flat,
                    parent_flat as i32,
                    slot_token(child),
                    slot_depth(child),
                    prng_seed,
                    root_position,
                );
            }
            emitted += 1;
            unsafe {
                *dfs_stack.add(stack_depth * 3) = child as u32;
                *dfs_stack.add(stack_depth * 3 + 1) = flat as u32;
                *dfs_stack.add(stack_depth * 3 + 2) = 0;
            }
            stack_depth += 1;
        } else {
            unsafe {
                (*nodes.add(parent_flat as usize)).trie_end = (emitted - 1) as u32;
            }
            stack_depth -= 1;
        }
    }

    for i in emitted..budget {
        unsafe { pad_node(nodes, token_ids, seeds, i) };
    }
    unsafe {
        *count = emitted as u32;
    }
}
