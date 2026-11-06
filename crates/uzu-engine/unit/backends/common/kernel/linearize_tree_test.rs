use std::mem::size_of;

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::{
                trie::TrieNode,
                weaver::{FRONTIER_NO_WINNER, FrontierIdx, TreeIdx},
            },
            kernel::{LinearizeChainKernel, LinearizeTreeKernel},
        },
        cpu::Cpu,
    },
    encodable_block::sampling::PRng,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const PRNG_SEED: u64 = 0x1234_5678_9abc_def0;
const ROOT_POSITION: u32 = 100;
/// Depth marker for padding rows, whose seeds are zeroed rather than derived.
const PAD_DEPTH: u32 = u32::MAX;

struct TreeSlot {
    token: u32,
    parent: u32,
    depth: u32,
    logprob: f32,
    valid: bool,
}

struct FrontierSlot {
    token: u32,
    parent: u32,
    depth: u32,
    logprob: f32,
    active: bool,
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

fn pack(
    tree: &[TreeSlot],
    frontier: &[FrontierSlot],
) -> (Vec<u32>, Vec<u32>, u32, u32) {
    let tree_slot_count = tree.len();
    let frontier_capacity = frontier.len();
    let mut packed_tree = vec![0u32; TreeIdx::COUNT * tree_slot_count];
    for (slot, s) in tree.iter().enumerate() {
        packed_tree[TreeIdx::TokenId as usize * tree_slot_count + slot] = s.token;
        packed_tree[TreeIdx::ParentSlot as usize * tree_slot_count + slot] = s.parent;
        packed_tree[TreeIdx::Depth as usize * tree_slot_count + slot] = s.depth;
        packed_tree[TreeIdx::PathLogprobBits as usize * tree_slot_count + slot] = s.logprob.to_bits();
        packed_tree[TreeIdx::Valid as usize * tree_slot_count + slot] = s.valid as u32;
    }
    // Keep the buffer nonzero even for an empty frontier; no slot is read when capacity is 0.
    let mut packed_frontier = vec![0u32; FrontierIdx::COUNT * frontier_capacity.max(1)];
    for (slot, s) in frontier.iter().enumerate() {
        packed_frontier[FrontierIdx::TokenId as usize * frontier_capacity + slot] = s.token;
        packed_frontier[FrontierIdx::ParentSlot as usize * frontier_capacity + slot] = s.parent;
        packed_frontier[FrontierIdx::Depth as usize * frontier_capacity + slot] = s.depth;
        packed_frontier[FrontierIdx::PathLogprobBits as usize * frontier_capacity + slot] = s.logprob.to_bits();
        packed_frontier[FrontierIdx::Active as usize * frontier_capacity + slot] = s.active as u32;
    }
    (packed_tree, packed_frontier, tree_slot_count as u32, frontier_capacity as u32)
}

fn run_tree<B: Backend>(
    tree: &[TreeSlot],
    frontier: &[FrontierSlot],
    budget: u32,
    with_seeds: bool,
) -> Linearized {
    let context = create_context::<B>();
    let kernel = <<B as Backend>::Kernels as Kernels>::LinearizeTreeKernel::new(&context, with_seeds).expect("kernel");
    let (packed_tree, packed_frontier, tree_slot_count, frontier_capacity) = pack(tree, frontier);
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

/// Tree with an invalid slot that has the best logprob and must be ignored,
/// plus frontier slots whose parents live in the tree or in the frontier.
/// Combined slot indices: tree slots 0..6, frontier slots 6..9.
fn main_tree() -> (Vec<TreeSlot>, Vec<FrontierSlot>) {
    let tree = vec![
        TreeSlot {
            token: 100,
            parent: FRONTIER_NO_WINNER,
            depth: 0,
            logprob: -0.1,
            valid: true,
        },
        TreeSlot {
            token: 101,
            parent: 0,
            depth: 1,
            logprob: -0.5,
            valid: true,
        },
        TreeSlot {
            token: 102,
            parent: 0,
            depth: 1,
            logprob: -0.2,
            valid: true,
        },
        TreeSlot {
            token: 103,
            parent: 1,
            depth: 2,
            logprob: -0.3,
            valid: true,
        },
        TreeSlot {
            token: 104,
            parent: 2,
            depth: 2,
            logprob: -0.05,
            valid: true,
        },
        TreeSlot {
            token: 999,
            parent: 0,
            depth: 1,
            logprob: -0.01,
            valid: false,
        },
    ];
    let frontier = vec![
        FrontierSlot {
            token: 105,
            parent: 1,
            depth: 2,
            logprob: -0.4,
            active: true,
        },
        FrontierSlot {
            token: 106,
            parent: 4,
            depth: 3,
            logprob: -0.6,
            active: true,
        },
        FrontierSlot {
            token: 107,
            parent: 6,
            depth: 3,
            logprob: -0.55,
            active: true,
        },
    ];
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
        assert_linearized_eq(&expected, &run_chain::<Cpu>(sampled, target, chain_length, budget, with_seeds), &label);
        for_each_non_cpu_backend!(|B| {
            assert_linearized_eq(&expected, &run_chain::<B>(sampled, target, chain_length, budget, with_seeds), &label);
        });
    }
}

#[uzu_test]
fn linearize_tree_selects_by_logprob_and_linearizes_preorder() {
    let (tree, frontier) = main_tree();
    for budget in [3u32, 5, 7, 9] {
        for with_seeds in [true, false] {
            let label = format!("budget={budget} seeds={with_seeds}");
            let expected = expected_main_tree(budget, with_seeds);
            assert_linearized_eq(&expected, &run_tree::<Cpu>(&tree, &frontier, budget, with_seeds), &label);
            for_each_non_cpu_backend!(|B| {
                assert_linearized_eq(&expected, &run_tree::<B>(&tree, &frontier, budget, with_seeds), &label);
            });
        }
    }
}

#[uzu_test]
fn linearize_tree_tie_breaks_to_smallest_slot() {
    let tree = vec![
        TreeSlot {
            token: 200,
            parent: FRONTIER_NO_WINNER,
            depth: 0,
            logprob: 0.0,
            valid: true,
        },
        TreeSlot {
            token: 201,
            parent: 0,
            depth: 1,
            logprob: -0.2,
            valid: true,
        },
        TreeSlot {
            token: 202,
            parent: 0,
            depth: 1,
            logprob: -0.2,
            valid: true,
        },
    ];
    let frontier = Vec::new();
    for (budget, expected) in [(2u32, expected_main_tie(2)), (3, expected_main_tie(3))] {
        let label = format!("budget={budget}");
        assert_linearized_eq(&expected, &run_tree::<Cpu>(&tree, &frontier, budget, true), &label);
        for_each_non_cpu_backend!(|B| {
            assert_linearized_eq(&expected, &run_tree::<B>(&tree, &frontier, budget, true), &label);
        });
    }
}

fn expected_main_tie(budget: u32) -> Linearized {
    let (nodes, tokens, depths) = match budget {
        2 => (vec![node(0, 1, 0, -1), node(1, 1, 1, 0)], vec![200, 201], vec![0, 1]),
        3 => (vec![node(0, 2, 0, -1), node(1, 1, 1, 0), node(2, 2, 1, 0)], vec![200, 201, 202], vec![0, 1, 1]),
        _ => unreachable!(),
    };
    let seeds = seeds_for(&depths, true);
    (nodes, tokens, seeds)
}

#[uzu_test]
fn linearize_tree_keeps_first_open_slot_when_its_logprob_is_nan() {
    let tree = vec![
        TreeSlot {
            token: 300,
            parent: FRONTIER_NO_WINNER,
            depth: 0,
            logprob: 0.0,
            valid: true,
        },
        TreeSlot {
            token: 301,
            parent: 0,
            depth: 1,
            logprob: f32::NAN,
            valid: true,
        },
        TreeSlot {
            token: 302,
            parent: 0,
            depth: 1,
            logprob: -0.5,
            valid: true,
        },
    ];
    let frontier = Vec::new();
    for budget in [2u32, 3] {
        let label = format!("budget={budget}");
        let expected = match budget {
            2 => (vec![node(0, 1, 0, -1), node(1, 1, 1, 0)], vec![300, 301], seeds_for(&[0, 1], true)),
            3 => (
                vec![node(0, 2, 0, -1), node(1, 1, 1, 0), node(2, 2, 1, 0)],
                vec![300, 301, 302],
                seeds_for(&[0, 1, 1], true),
            ),
            _ => unreachable!(),
        };
        assert_linearized_eq(&expected, &run_tree::<Cpu>(&tree, &frontier, budget, true), &label);
        for_each_non_cpu_backend!(|B| {
            assert_linearized_eq(&expected, &run_tree::<B>(&tree, &frontier, budget, true), &label);
        });
    }
}
