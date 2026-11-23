use std::mem::size_of;

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        gpu_types::{
            trie::TrieNode,
            weaver::{FRONTIER_NO_WINNER, FrontierIdx, TreeIdx},
        },
        kernel::{LinearizeChainKernel, LinearizeTreeKernel},
    },
    encodable_block::sampling::PRng,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_backend},
};

const PRNG_SEED: u64 = 0x1234_5678_9abc_def0;
const ROOT_POSITION: u32 = 100;
const PAD_DEPTH: u32 = u32::MAX;

/// A tree or frontier slot; `live` is the tree's `Valid` or the frontier's `Active` flag.
struct Slot {
    token: u32,
    parent: u32,
    depth: u32,
    edge: f32,
    path: f32,
    live: bool,
}

fn slot(
    token: u32,
    parent: u32,
    depth: u32,
    edge: f32,
) -> Slot {
    Slot {
        token,
        parent,
        depth,
        edge,
        path: 0.0,
        live: true,
    }
}

fn node(
    trie_start: u32,
    trie_end: u32,
    height: u32,
    parent: i32,
) -> TrieNode {
    TrieNode {
        trie_start,
        trie_end,
        height,
        parent,
    }
}

fn padding() -> TrieNode {
    node(u32::MAX, 0, 0, -1)
}

fn as_words(nodes: &[TrieNode]) -> Vec<[u32; 4]> {
    bytemuck::cast_slice(nodes).to_vec()
}

fn seed(depth: u32) -> u64 {
    PRng::new(PRNG_SEED).derive(ROOT_POSITION as u64 + depth as u64)
}

fn seeds_for(
    depths: &[u32],
    with_seeds: bool,
) -> Vec<u64> {
    if with_seeds {
        depths
            .iter()
            .map(|&depth| {
                if depth == PAD_DEPTH {
                    0
                } else {
                    seed(depth)
                }
            })
            .collect()
    } else {
        Vec::new()
    }
}

type Linearized = (Vec<TrieNode>, Vec<u32>, Vec<u64>);

fn assert_linearized_eq(
    expected: &Linearized,
    actual: &Linearized,
    label: &str,
) {
    assert_eq!(as_words(&expected.0), as_words(&actual.0), "nodes {label}");
    assert_eq!(expected.1, actual.1, "token ids {label}");
    assert_eq!(expected.2, actual.2, "seeds {label}");
}

fn run_chain<B: Backend>(
    sampled: &[u32],
    target_output_token: u32,
    chain_length: u32,
    budget: u32,
    with_seeds: bool,
) -> Linearized {
    let context = create_context::<B>();
    let kernel = <<B as Backend>::Kernels as Kernels>::LinearizeChainKernel::new(&context, with_seeds).expect("kernel");
    let sampled = create_buffer_with_data::<B, u32>(&context, sampled);
    let target = create_buffer_with_data::<B, u32>(&context, &[target_output_token]);
    let mut nodes = create_buffer::<B, u32>(&context, budget as usize * size_of::<TrieNode>() / size_of::<u32>());
    let mut token_ids = create_buffer::<B, u32>(&context, budget as usize);
    let mut seeds = with_seeds.then(|| create_buffer::<B, u64>(&context, budget as usize));

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(
        &sampled,
        &target,
        &mut nodes,
        &mut token_ids,
        seeds.as_mut(),
        chain_length,
        budget,
        PRNG_SEED,
        ROOT_POSITION,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    let nodes = bytemuck::cast_slice::<u32, TrieNode>(&buffer_to_vec::<B, u32>(&*context, &nodes)).to_vec();
    let token_ids = buffer_to_vec::<B, u32>(&*context, &token_ids);
    let seeds = seeds.map(|seeds| buffer_to_vec::<B, u64>(&*context, &seeds)).unwrap_or_default();
    (nodes, token_ids, seeds)
}

fn expected_chain(
    sampled: &[u32],
    target: u32,
    chain_length: u32,
    budget: u32,
    with_seeds: bool,
) -> Linearized {
    let chain = (chain_length as usize + 1).min(budget as usize);
    let mut nodes = Vec::with_capacity(budget as usize);
    let mut tokens = Vec::with_capacity(budget as usize);
    let mut depths = Vec::with_capacity(budget as usize);
    for i in 0..budget as usize {
        if i < chain {
            nodes.push(node(
                i as u32,
                (chain - 1) as u32,
                i as u32,
                if i == 0 {
                    -1
                } else {
                    (i - 1) as i32
                },
            ));
            tokens.push(if i == 0 {
                target
            } else {
                sampled[i - 1]
            });
            depths.push(i as u32);
        } else {
            nodes.push(padding());
            tokens.push(0);
            depths.push(PAD_DEPTH);
        }
    }
    let seeds = seeds_for(&depths, with_seeds);
    (nodes, tokens, seeds)
}

fn pack_columns(
    slots: &[Slot],
    field_count: usize,
    [token, parent, depth, edge, path, live]: [usize; 6],
) -> Vec<u32> {
    let count = slots.len();
    // Keep the buffer nonzero even for no slots; no slot is read when the count is 0.
    let mut packed = vec![0u32; field_count * count.max(1)];
    for (i, s) in slots.iter().enumerate() {
        packed[token * count + i] = s.token;
        packed[parent * count + i] = s.parent;
        packed[depth * count + i] = s.depth;
        packed[edge * count + i] = s.edge.to_bits();
        packed[path * count + i] = s.path.to_bits();
        packed[live * count + i] = s.live as u32;
    }
    packed
}

fn run_tree<B: Backend>(
    tree: &[Slot],
    frontier: &[Slot],
    budget: u32,
    with_seeds: bool,
) -> Linearized {
    let context = create_context::<B>();
    let kernel = <<B as Backend>::Kernels as Kernels>::LinearizeTreeKernel::new(&context, with_seeds).expect("kernel");
    let packed_tree = pack_columns(
        tree,
        TreeIdx::COUNT,
        [
            TreeIdx::TokenId,
            TreeIdx::ParentSlot,
            TreeIdx::Depth,
            TreeIdx::EdgeLogprobBits,
            TreeIdx::PathLogprobBits,
            TreeIdx::Valid,
        ]
        .map(|i| i as usize),
    );
    let packed_frontier = pack_columns(
        frontier,
        FrontierIdx::COUNT,
        [
            FrontierIdx::TokenId,
            FrontierIdx::ParentSlot,
            FrontierIdx::Depth,
            FrontierIdx::EdgeLogprobBits,
            FrontierIdx::PathLogprobBits,
            FrontierIdx::Active,
        ]
        .map(|i| i as usize),
    );
    let tree_slot_count = tree.len() as u32;
    let frontier_capacity = frontier.len() as u32;
    let packed_tree = create_buffer_with_data::<B, u32>(&context, &packed_tree);
    let frontier_buffer = create_buffer_with_data::<B, u32>(&context, &packed_frontier);
    let total_slots = tree_slot_count as usize + frontier_capacity as usize;
    let max_depth = tree.iter().map(|s| s.depth).chain(frontier.iter().map(|s| s.depth)).max().unwrap_or(0);
    let mut nodes = create_buffer::<B, u32>(&context, budget as usize * size_of::<TrieNode>() / size_of::<u32>());
    let mut token_ids = create_buffer::<B, u32>(&context, budget as usize);
    let mut seeds = with_seeds.then(|| create_buffer::<B, u64>(&context, budget as usize));
    let mut stack = create_buffer::<B, u32>(&context, total_slots + 3 * (max_depth as usize + 2));

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(
        &packed_tree,
        &frontier_buffer,
        &mut nodes,
        &mut token_ids,
        seeds.as_mut(),
        &mut stack,
        tree_slot_count,
        frontier_capacity,
        budget,
        PRNG_SEED,
        ROOT_POSITION,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    let nodes = bytemuck::cast_slice::<u32, TrieNode>(&buffer_to_vec::<B, u32>(&*context, &nodes)).to_vec();
    let token_ids = buffer_to_vec::<B, u32>(&*context, &token_ids);
    let seeds = seeds.map(|seeds| buffer_to_vec::<B, u64>(&*context, &seeds)).unwrap_or_default();
    (nodes, token_ids, seeds)
}

/// Tree with an invalid slot that has the best edge logprob and must be ignored,
/// plus frontier slots whose parents live in the tree or in the frontier.
/// Combined slot indices: tree slots 0..6, frontier slots 6..9.
/// Edges are dyadic so the summed scores are exact: slot 1 -0.5, 2 -0.25, 3 -0.625, 4 -0.375,
/// 6 -0.75, 7 -0.875, 8 -0.8125.
fn main_tree() -> (Vec<Slot>, Vec<Slot>) {
    let tree = vec![
        slot(100, FRONTIER_NO_WINNER, 0, 0.0),
        slot(101, 0, 1, -0.5),
        slot(102, 0, 1, -0.25),
        slot(103, 1, 2, -0.125),
        slot(104, 2, 2, -0.125),
        Slot {
            live: false,
            ..slot(999, 0, 1, -0.01)
        },
    ];
    let frontier = vec![slot(105, 1, 2, -0.25), slot(106, 4, 3, -0.5), slot(107, 6, 3, -0.0625)];
    (tree, frontier)
}

fn expected_main_tree(
    budget: u32,
    with_seeds: bool,
) -> Linearized {
    let (nodes, tokens, depths): (Vec<TrieNode>, Vec<u32>, Vec<u32>) = match budget {
        // Kept: root, slot 2, slot 4. Preorder: 0, 2, 4.
        3 => (vec![node(0, 2, 0, -1), node(1, 2, 1, 0), node(2, 2, 2, 1)], vec![100, 102, 104], vec![0, 1, 2]),
        // Kept: 0, 1, 2, 3, 4. Preorder: 0, 1, 3, 2, 4.
        5 => (
            vec![node(0, 4, 0, -1), node(1, 2, 1, 0), node(2, 2, 2, 1), node(3, 4, 1, 0), node(4, 4, 2, 3)],
            vec![100, 101, 103, 102, 104],
            vec![0, 1, 2, 1, 2],
        ),
        // Kept: 0, 1, 2, 3, 4, frontier 0 (slot 6), frontier 2 (slot 8, child of slot 6).
        // Preorder: 0, 1, 3, 6, 8, 2, 4.
        7 => (
            vec![
                node(0, 6, 0, -1),
                node(1, 4, 1, 0),
                node(2, 2, 2, 1),
                node(3, 4, 2, 1),
                node(4, 4, 3, 3),
                node(5, 6, 1, 0),
                node(6, 6, 2, 5),
            ],
            vec![100, 101, 103, 105, 107, 102, 104],
            vec![0, 1, 2, 2, 3, 1, 2],
        ),
        // All 8 valid slots kept; one padding row.
        // Preorder: 0, 1, 3, 6, 8, 2, 4, 7.
        9 => (
            vec![
                node(0, 7, 0, -1),
                node(1, 4, 1, 0),
                node(2, 2, 2, 1),
                node(3, 4, 2, 1),
                node(4, 4, 3, 3),
                node(5, 7, 1, 0),
                node(6, 7, 2, 5),
                node(7, 7, 3, 6),
                padding(),
            ],
            vec![100, 101, 103, 105, 107, 102, 104, 106, 0],
            vec![0, 1, 2, 2, 3, 1, 2, 3, PAD_DEPTH],
        ),
        _ => unreachable!(),
    };
    let seeds = seeds_for(&depths, with_seeds);
    (nodes, tokens, seeds)
}

fn assert_tree_on_all_backends(
    tree: &[Slot],
    frontier: &[Slot],
    budget: u32,
    with_seeds: bool,
    expected: &Linearized,
) {
    let label = format!("budget={budget} seeds={with_seeds}");
    for_each_backend!(|B| {
        assert_linearized_eq(expected, &run_tree::<B>(tree, frontier, budget, with_seeds), &label);
    });
}

fn assert_first_child_kept_alone(
    first_edge: f32,
    second_edge: f32,
) {
    let tree = [slot(200, FRONTIER_NO_WINNER, 0, 0.0), slot(201, 0, 1, first_edge), slot(202, 0, 1, second_edge)];
    let first_only = (vec![node(0, 1, 0, -1), node(1, 1, 1, 0)], vec![200, 201], seeds_for(&[0, 1], true));
    let both =
        (vec![node(0, 2, 0, -1), node(1, 1, 1, 0), node(2, 2, 1, 0)], vec![200, 201, 202], seeds_for(&[0, 1, 1], true));
    assert_tree_on_all_backends(&tree, &[], 2, true, &first_only);
    assert_tree_on_all_backends(&tree, &[], 3, true, &both);
}

#[uzu_test]
fn linearize_chain_linearizes_and_pads() {
    let cases: &[(&[u32], u32, u32, u32, bool)] = &[
        (&[11, 22, 33], 7, 3, 6, true),
        (&[11, 22, 33], 7, 10, 4, true), // chain clamped to budget
        (&[11, 22, 33], 7, 3, 6, false),
        (&[5], 9, 0, 3, true), // chain of just the root
    ];
    for &(sampled, target, chain_length, budget, with_seeds) in cases {
        let label =
            format!("sampled={sampled:?} target={target} chain={chain_length} budget={budget} seeds={with_seeds}");
        let expected = expected_chain(sampled, target, chain_length, budget, with_seeds);
        for_each_backend!(|B| {
            assert_linearized_eq(&expected, &run_chain::<B>(sampled, target, chain_length, budget, with_seeds), &label);
        });
    }
}

#[uzu_test]
fn linearize_tree_selects_by_summed_edges_and_linearizes_preorder() {
    let (tree, frontier) = main_tree();
    for budget in [3u32, 5, 7, 9] {
        for with_seeds in [true, false] {
            assert_tree_on_all_backends(&tree, &frontier, budget, with_seeds, &expected_main_tree(budget, with_seeds));
        }
    }
}

/// Pruning ranks by the summed edge logprobs, which carry the target's Gumbel noise (or the confidence head's
/// refinement), not by the clean path logprobs the weaver expanded by. Here the two disagree, and ranking by a
/// slot's own edge instead of the root-to-slot sum would also disagree at budget 4.
/// Combined slot indices: tree slots 0..5, frontier slot 5.
#[uzu_test]
fn linearize_tree_ranks_by_summed_edges_not_path() {
    let with_path = |path: f32, slot: Slot| Slot {
        path,
        ..slot
    };
    let tree = [
        slot(300, FRONTIER_NO_WINNER, 0, 0.0),
        with_path(-0.25, slot(301, 0, 1, -1.0)), // score -1.0
        with_path(-1.0, slot(302, 0, 1, -0.25)), // score -0.25
        with_path(-0.5, slot(303, 1, 2, -0.25)), // score -1.25
        with_path(-1.5, slot(304, 2, 2, -0.5)),  // score -0.75
    ];
    let frontier = [with_path(-2.0, slot(305, 4, 3, -0.375))]; // score -1.125

    // Kept: 0, 2, 4 (by path it would be 0, 1, 3). Preorder: 0, 2, 4.
    let budget_3 =
        (vec![node(0, 2, 0, -1), node(1, 2, 1, 0), node(2, 2, 2, 1)], vec![300, 302, 304], seeds_for(&[0, 1, 2], true));
    // Kept: 0, 1, 2, 4 (by own edge it would be 0, 2, 4, 5). Preorder: 0, 1, 2, 4.
    let budget_4 = (
        vec![node(0, 3, 0, -1), node(1, 1, 1, 0), node(2, 3, 1, 0), node(3, 3, 2, 2)],
        vec![300, 301, 302, 304],
        seeds_for(&[0, 1, 1, 2], true),
    );
    assert_tree_on_all_backends(&tree, &frontier, 3, true, &budget_3);
    assert_tree_on_all_backends(&tree, &frontier, 4, true, &budget_4);
}

#[uzu_test]
fn linearize_tree_tie_breaks_to_smallest_slot() {
    assert_first_child_kept_alone(-0.2, -0.2);
}

#[uzu_test]
fn linearize_tree_keeps_first_open_slot_when_its_score_is_nan() {
    assert_first_child_kept_alone(f32::NAN, -0.5);
}
