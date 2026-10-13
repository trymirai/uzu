use uzu_engine_macros::kernel;

use crate::backends::common::gpu_types::weaver::{FrontierIdx, TreeIdx};

const NONE: u32 = u32::MAX;

/// The tokens of the head's tree attention: the root, then one token per frontier entry (entry slot * C + rank is
/// that slot's child of that rank, kept by the frontier after the child is expanded into a slot of its own). A token
/// is present when its child is real. Its row is the depth embedding, plus the child's projected vector, plus the
/// projected z_u of the slot it was expanded into when that slot has real children of its own. The metadata the
/// attention and the prune read: parent token, depth, presence, slot, the ancestor token at every depth, and the
/// token of every slot.
#[kernel(ConfidenceHeadTreeTokens)]
pub fn confidence_head_tree_tokens(
    packed_tree: *const u32,
    frontier: *const u32,
    child_real: *const u32,
    node_inputs: *const f32,
    child_inputs: *const f32,
    depth_embedding: *const f32,
    tokens: *mut f32,
    token_parents: *mut u32,
    token_depths: *mut u32,
    token_present: *mut u32,
    token_slots: *mut u32,
    token_ancestors: *mut u32,
    slot_tokens: *mut u32,
    slot_count: u32,
    children: u32,
    width: u32,
    max_depth: u32,
) {
    let (slots, children, width, max_depth) =
        (slot_count as usize, children as usize, width as usize, max_depth as usize);
    let entries = slots * children;
    let tree = |field: TreeIdx, slot: usize| unsafe { *packed_tree.add(field as usize * slots + slot) };
    let entry = |field: FrontierIdx, index: usize| unsafe { *frontier.add(field as usize * entries + index) };
    let real = |index: usize| unsafe { *child_real.add(index) } != 0;
    let has_children = |slot: usize| (0..children).any(|child| real(slot * children + child));
    // The token of a slot: the root's, or that of its entry under its parent (found by its token, pool tokens being
    // distinct among a parent's children).
    let slot_token = |slot: usize| -> u32 {
        if slot == 0 {
            return 0;
        }
        let parent = tree(TreeIdx::ParentSlot, slot) as usize;
        (0..children)
            .map(|child| parent * children + child)
            .find(|&index| real(index) && entry(FrontierIdx::TokenId, index) == tree(TreeIdx::TokenId, slot))
            .map_or(NONE, |index| 1 + index as u32)
    };
    for token in 0..1 + entries {
        let (present, depth, parent_slot, slot) = if token == 0 {
            (tree(TreeIdx::Valid, 0) != 0, 0usize, None, Some(0usize))
        } else {
            let index = token - 1;
            let parent = index / children;
            if !real(index) {
                (false, 0, None, None)
            } else {
                let token_id = entry(FrontierIdx::TokenId, index);
                let slot = (1..slots).find(|&slot| {
                    tree(TreeIdx::Valid, slot) != 0
                        && tree(TreeIdx::ParentSlot, slot) as usize == parent
                        && tree(TreeIdx::TokenId, slot) == token_id
                });
                (true, entry(FrontierIdx::Depth, index) as usize, Some(parent), slot)
            }
        };
        let mut ancestors = vec![NONE; max_depth + 1];
        if present {
            ancestors[depth.min(max_depth)] = token as u32;
            let mut walk = parent_slot;
            while let Some(slot) = walk {
                ancestors[(tree(TreeIdx::Depth, slot) as usize).min(max_depth)] = slot_token(slot);
                walk = (slot != 0).then(|| tree(TreeIdx::ParentSlot, slot) as usize);
            }
        }
        let mut row = vec![0.0f32; width];
        if present {
            let embedding = depth.min(max_depth) * width;
            for unit in 0..width {
                row[unit] = unsafe { *depth_embedding.add(embedding + unit) };
            }
            if token > 0 {
                for unit in 0..width {
                    row[unit] += unsafe { *child_inputs.add((token - 1) * width + unit) };
                }
            }
            if let Some(slot) = slot.filter(|&slot| has_children(slot)) {
                for unit in 0..width {
                    row[unit] += unsafe { *node_inputs.add(slot * width + unit) };
                }
            }
        }
        unsafe {
            for (unit, value) in row.into_iter().enumerate() {
                *tokens.add(token * width + unit) = value;
            }
            *token_parents.add(token) = parent_slot.map_or(NONE, slot_token);
            *token_depths.add(token) = depth as u32;
            *token_present.add(token) = present as u32;
            *token_slots.add(token) = slot.map_or(NONE, |slot| slot as u32);
            for (level, ancestor) in ancestors.into_iter().enumerate() {
                *token_ancestors.add(token * (max_depth + 1) + level) = ancestor;
            }
            if let Some(slot) = slot.filter(|_| present) {
                *slot_tokens.add(slot) = token as u32;
            }
        }
    }
}
