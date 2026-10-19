use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx},
            kernel::WeaverTopChildrenKernel,
        },
        cpu::Cpu,
    },
    encodable_block::sampling::{gumbel_float, revidx},
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const CANDIDATES: usize = 512;
const CHILDREN: usize = 8;
const VOCAB_SIZE: u32 = 131_072;
const DEPTH_SEEDS: [u64; 3] = [0x9E3779B97F4A7C15, 0xD1B54A32D192ED03, 0x2545F4914F6CDD1D];
const PRUNE_NOISE_SCALE: f32 = 1.0 / 1.5;
// Rows expand tree slots 2, 0 and 1 (row 1 is padding and expands nothing).
const ROW_TREE_SLOTS: [u32; 3] = [2, 0, 1];
const ROW_DEPTHS: [u32; 3] = [1, 0, 2];
const ROW_VALID: [u32; 3] = [1, 0, 1];
const TREE_SLOTS: usize = 3;
const TREE_PATH_LOGPROBS: [f32; TREE_SLOTS] = [0.0, -1.5, -0.25];
const TREE_DEPTHS: [u32; TREE_SLOTS] = [0, 2, 1];
const FRONTIER_CAPACITY: usize = TREE_SLOTS * CHILDREN;
const UNWRITTEN: u32 = 42;

struct Inputs {
    residual: Vec<bf16>,
    candidate_logits: Vec<f32>,
    ids: Vec<u32>,
}

fn inputs() -> Inputs {
    let rows = ROW_TREE_SLOTS.len();
    Inputs {
        residual: (0..rows * CANDIDATES)
            .map(|index| bf16::from_f32(((index as f32 * 0.017).cos() * 4.0).round() * 0.125))
            .collect(),
        candidate_logits: (0..rows * CANDIDATES)
            .map(|index| ((index as f32 * 0.011).sin() * 3.0).round() * 0.125)
            .collect(),
        ids: (0..rows)
            .flat_map(|row| (0..CANDIDATES).rev().map(move |index| 70_000 + (row * CANDIDATES + index) as u32))
            .collect(),
    }
}

fn top_children<B: Backend>(
    inputs: &Inputs,
    expansion_candidates: usize,
    prune_noise_scale: Option<f32>,
) -> Vec<u32> {
    let rows = ROW_TREE_SLOTS.len();
    let context = create_context::<B>();
    let mut metadata = vec![0u32; rows * MetadataIdx::COUNT];
    for row in 0..rows {
        metadata[MetadataIdx::TreeSlot as usize * rows + row] = ROW_TREE_SLOTS[row];
        metadata[MetadataIdx::Depth as usize * rows + row] = ROW_DEPTHS[row];
    }
    let mut tree = vec![0u32; TreeIdx::COUNT * TREE_SLOTS];
    for slot in 0..TREE_SLOTS {
        tree[TreeIdx::PathLogprobBits as usize * TREE_SLOTS + slot] = TREE_PATH_LOGPROBS[slot].to_bits();
        tree[TreeIdx::Depth as usize * TREE_SLOTS + slot] = TREE_DEPTHS[slot];
    }
    // The CPU backend runs the dispatch at submit, so every buffer must outlive the wait.
    let residual = create_buffer_with_data::<B, bf16>(&context, &inputs.residual);
    let candidate_logits = create_buffer_with_data::<B, f32>(&context, &inputs.candidate_logits);
    let ids = create_buffer_with_data::<B, u32>(&context, &inputs.ids);
    let depth_seeds = create_buffer_with_data::<B, u64>(&context, &DEPTH_SEEDS);
    let metadata = create_buffer_with_data::<B, u32>(&context, &metadata);
    let valid = create_buffer_with_data::<B, u32>(&context, &ROW_VALID);
    let tree = create_buffer_with_data::<B, u32>(&context, &tree);
    let mut frontier =
        create_buffer_with_data::<B, u32>(&context, &vec![UNWRITTEN; FrontierIdx::COUNT * FRONTIER_CAPACITY]);
    let kernel = <B::Kernels as Kernels>::WeaverTopChildrenKernel::new(&context, prune_noise_scale.is_some()).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &residual,
        &candidate_logits,
        &ids,
        &depth_seeds,
        &metadata,
        &valid,
        &tree,
        &mut frontier,
        rows as u32,
        CANDIDATES as u32,
        expansion_candidates as u32,
        CHILDREN as u32,
        VOCAB_SIZE,
        FRONTIER_CAPACITY as u32,
        TREE_SLOTS as u32,
        prune_noise_scale,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec(&frontier)
}

fn lane(
    frontier: &[u32],
    field: FrontierIdx,
    slot: usize,
) -> u32 {
    frontier[field as usize * FRONTIER_CAPACITY + slot]
}

// Candidate index of each child of `row`, in child order.
fn child_indices(
    inputs: &Inputs,
    frontier: &[u32],
    row: usize,
) -> Vec<usize> {
    let parent = ROW_TREE_SLOTS[row] as usize;
    (0..CHILDREN)
        .map(|child| {
            let token = lane(frontier, FrontierIdx::TokenId, parent * CHILDREN + child);
            (0..CANDIDATES).find(|&index| inputs.ids[row * CANDIDATES + index] == token).unwrap()
        })
        .collect()
}

#[uzu_test]
fn weaver_top_children_matches_cpu() {
    let inputs = inputs();
    for expansion_candidates in [CANDIDATES, 32] {
        for prune_noise_scale in [None, Some(PRUNE_NOISE_SCALE)] {
            let expected = top_children::<Cpu>(&inputs, expansion_candidates, prune_noise_scale);
            for_each_non_cpu_backend!(|B| {
                let actual = top_children::<B>(&inputs, expansion_candidates, prune_noise_scale);
                for field in [FrontierIdx::TokenId, FrontierIdx::ParentSlot, FrontierIdx::Depth, FrontierIdx::Active] {
                    for slot in 0..FRONTIER_CAPACITY {
                        assert_eq!(lane(&actual, field, slot), lane(&expected, field, slot));
                    }
                }
                // The edge lane also carries the Gumbel transform, which Metal evaluates with fast-math logs.
                for (field, tolerance) in [(FrontierIdx::PathLogprobBits, 1e-5), (FrontierIdx::EdgeLogprobBits, 1e-3)] {
                    for slot in 0..FRONTIER_CAPACITY {
                        let actual = f32::from_bits(lane(&actual, field, slot));
                        let expected = f32::from_bits(lane(&expected, field, slot));
                        assert!((actual - expected).abs() < tolerance, "lane {} slot {slot}", field as usize);
                    }
                }
            });
        }
    }
}

/// Children of a node go to frontier slots parent * expand_width + child with the path logprob (parent path plus the
/// pool log-softmax of the child), and with prune noise the edge lane carries the log-softmax of logits plus the
/// target sampler's noise at the node's depth seed, scaled by `prune_noise_scale`. The reference takes that noise from
/// `gumbel_float` on purpose, since matching the target sampler is the contract; the seed, the token indexing and the
/// log-sum-exp are computed here independently, in f64. Padding rows write nothing.
#[uzu_test]
fn weaver_top_children_frontier_matches_reference() {
    let inputs = inputs();
    let model = top_children::<Cpu>(&inputs, CANDIDATES, None);
    let prune = top_children::<Cpu>(&inputs, CANDIDATES, Some(PRUNE_NOISE_SCALE));
    let log_softmax = |values: &[f64], index: usize| {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        values[index] - (values.iter().map(|value| (value - max).exp()).sum::<f64>().ln() + max)
    };
    for row in 0..ROW_TREE_SLOTS.len() {
        let parent = ROW_TREE_SLOTS[row] as usize;
        let slots = parent * CHILDREN..(parent + 1) * CHILDREN;
        if ROW_VALID[row] == 0 {
            for frontier in [&model, &prune] {
                assert!(slots.clone().all(|slot| {
                    (0..FrontierIdx::COUNT).all(|field| frontier[field * FRONTIER_CAPACITY + slot] == UNWRITTEN)
                }));
            }
            continue;
        }
        let logits = (0..CANDIDATES)
            .map(|index| {
                let flat = row * CANDIDATES + index;
                inputs.candidate_logits[flat] as f64 + inputs.residual[flat].to_f64()
            })
            .collect::<Vec<_>>();
        let seed = DEPTH_SEEDS[ROW_DEPTHS[row] as usize];
        let noisy = (0..CANDIDATES)
            .map(|index| {
                let token = inputs.ids[row * CANDIDATES + index];
                logits[index] + PRUNE_NOISE_SCALE as f64 * gumbel_float(seed, revidx(token, VOCAB_SIZE)) as f64
            })
            .collect::<Vec<_>>();
        for frontier in [&model, &prune] {
            for (child, index) in child_indices(&inputs, frontier, row).into_iter().enumerate() {
                let slot = parent * CHILDREN + child;
                assert_eq!(lane(frontier, FrontierIdx::ParentSlot, slot), parent as u32);
                assert_eq!(lane(frontier, FrontierIdx::Depth, slot), TREE_DEPTHS[parent] + 1);
                assert_eq!(lane(frontier, FrontierIdx::Active, slot), 1);
                let path = f32::from_bits(lane(frontier, FrontierIdx::PathLogprobBits, slot)) as f64;
                // An f32 log-sum-exp over 512 values of magnitude below 20 keeps about 1e-4 of absolute accuracy.
                assert!((path - (TREE_PATH_LOGPROBS[parent] as f64 + log_softmax(&logits, index))).abs() < 1e-4);
            }
        }
        for (child, index) in child_indices(&inputs, &prune, row).into_iter().enumerate() {
            let edge = f32::from_bits(lane(&prune, FrontierIdx::EdgeLogprobBits, parent * CHILDREN + child)) as f64;
            assert!((edge - log_softmax(&noisy, index)).abs() < 1e-4);
        }
    }
}

/// With `expansion_candidates` below the pool size, children come only from the pool's first candidates (the pool is
/// sorted by draft logit), while the unrestricted expansion of the same rows reaches past them.
#[uzu_test]
fn weaver_top_children_expands_only_leading_candidates() {
    const EXPANSION_CANDIDATES: usize = 32;
    let inputs = inputs();
    let restricted = top_children::<Cpu>(&inputs, EXPANSION_CANDIDATES, Some(PRUNE_NOISE_SCALE));
    let unrestricted = top_children::<Cpu>(&inputs, CANDIDATES, Some(PRUNE_NOISE_SCALE));
    let valid_rows = (0..ROW_TREE_SLOTS.len()).filter(|&row| ROW_VALID[row] == 1).collect::<Vec<_>>();
    assert!(valid_rows.iter().all(|&row| {
        child_indices(&inputs, &restricted, row).into_iter().all(|index| index < EXPANSION_CANDIDATES)
    }));
    assert!(valid_rows.iter().any(|&row| {
        child_indices(&inputs, &unrestricted, row).into_iter().any(|index| index >= EXPANSION_CANDIDATES)
    }));
}
