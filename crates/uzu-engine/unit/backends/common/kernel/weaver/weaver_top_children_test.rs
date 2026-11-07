use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::weaver::{
                FrontierIdx, MetadataIdx, TreeIdx,
                WeaverSelectionMode::{self, Children, Fused},
            },
            kernel::WeaverSelectKernel,
        },
        cpu::Cpu,
    },
    encodable_block::sampling::{gumbel_float, revidx},
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const CANDIDATES: usize = 512;
const CHILDREN: usize = 8;
const VOCAB_SIZE: u32 = 131_072;
const DEPTHS: [u32; 3] = [0, 1, 2];
const DEPTH_SEEDS: [u64; 3] = [0x9E3779B97F4A7C15, 0xD1B54A32D192ED03, 0x2545F4914F6CDD1D];
const ROWS: usize = 3;
/// Fused selection takes the next batch, as many nodes as rows, into the tree slots after the rows'.
const NODES: usize = ROWS;
const TREE_SLOT_COUNT: usize = ROWS + 1 + NODES;
const FRONTIER_CAPACITY: usize = (ROWS + 1) * CHILDREN;
const ANCESTOR_STRIDE: usize = 4;
/// Fused selection expands the nodes shallower than this.
const LOOKAHEAD_COUNT: u32 = 3;
const PRUNE_NOISE_SCALE: f32 = 1.0 / 1.5;
const FUSED_TOLERANCE: f32 = 1e-5;

fn inputs() -> (Vec<bf16>, Vec<f32>, Vec<u32>) {
    let residual = (0..ROWS * CANDIDATES)
        .map(|index| bf16::from_f32(((index as f32 * 0.017).cos() * 4.0).round() * 0.125))
        .collect::<Vec<_>>();
    let candidate_logits =
        (0..ROWS * CANDIDATES).map(|index| ((index as f32 * 0.011).sin() * 3.0).round() * 0.125).collect::<Vec<_>>();
    let ids = (0..ROWS)
        .flat_map(|row| (0..CANDIDATES).rev().map(move |index| 70_000 + (row * CANDIDATES + index) as u32))
        .collect::<Vec<_>>();
    (residual, candidate_logits, ids)
}

fn score_key(bits: u32) -> u32 {
    if bits & 0x8000_0000 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

/// The frontier, the tree and, for `Fused`, the next batch's slot ancestors, tokens, metadata, ancestor indices,
/// validity, candidate ids and candidate logit bits. Fused selection reads the rows' children from the nodes'
/// candidates, as after the first round, and gives the nodes shallower than ROWS their depth's pool candidates.
fn run_select<B: Backend>(
    mode: WeaverSelectionMode,
    children_noise_scale: f32,
    prune_noise_scale: Option<f32>,
    node_valid: &[u32; ROWS],
    tree_values: &[u32; TreeIdx::COUNT * TREE_SLOT_COUNT],
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let fused = mode == Fused;
    let (residual, candidate_logits, ids) = inputs();
    let context = create_context::<B>();
    let residual = create_buffer_with_data::<B, bf16>(&context, &residual);
    let pool_logits = create_buffer_with_data::<B, f32>(&context, &candidate_logits);
    let mut candidate_logits = create_buffer_with_data::<B, f32>(&context, &candidate_logits);
    let pool_ids = create_buffer_with_data::<B, u32>(&context, &ids);
    let mut ids = create_buffer_with_data::<B, u32>(&context, &ids);
    let depth_seeds = create_buffer_with_data::<B, u64>(&context, &DEPTH_SEEDS);
    let mut metadata_values = vec![0u32; ROWS * MetadataIdx::COUNT];
    for row in 0..ROWS {
        metadata_values[MetadataIdx::Depth as usize * ROWS + row] = DEPTHS[row];
        metadata_values[MetadataIdx::TreeSlot as usize * ROWS + row] = row as u32 + 1;
    }
    let mut node_metadata = create_buffer_with_data::<B, u32>(&context, &metadata_values);
    let mut tree = create_buffer_with_data::<B, u32>(&context, tree_values);
    let mut valid = create_buffer_with_data::<B, u32>(&context, node_valid);
    let mut frontier_values = vec![42; FrontierIdx::COUNT * FRONTIER_CAPACITY];
    if fused {
        frontier_values[FrontierIdx::Active as usize * FRONTIER_CAPACITY..].fill(0);
    }
    let mut frontier = create_buffer_with_data::<B, u32>(&context, &frontier_values);
    let mut slot_ancestors = create_buffer_with_data::<B, u32>(
        &context,
        &(0..(TREE_SLOT_COUNT * ANCESTOR_STRIDE) as u32).collect::<Vec<_>>(),
    );
    let mut token_ids = create_buffer_with_data::<B, u32>(&context, &[7; NODES]);
    let mut ancestor_indices = create_buffer_with_data::<B, u32>(&context, &[7; NODES * ANCESTOR_STRIDE]);
    let block_threads = if fused {
        1024
    } else {
        256
    };
    let kernel =
        <B::Kernels as Kernels>::WeaverSelectKernel::new(&context, block_threads, mode, prune_noise_scale.is_some())
            .unwrap();
    let next_batch = |value: usize| u32::from(fused) * value as u32;
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        Some(&residual),
        Some(&depth_seeds),
        &mut frontier,
        &mut tree,
        fused.then_some(&mut slot_ancestors),
        fused.then_some(&mut token_ids),
        &mut node_metadata,
        fused.then_some(&mut ancestor_indices),
        &mut valid,
        &pool_ids,
        &pool_logits,
        &mut ids,
        &mut candidate_logits,
        ROWS as u32,
        u32::from(!fused),
        CANDIDATES as u32,
        CHILDREN as u32,
        VOCAB_SIZE,
        FRONTIER_CAPACITY as u32,
        TREE_SLOT_COUNT as u32,
        next_batch(NODES),
        next_batch(ROWS + 1),
        next_batch(ANCESTOR_STRIDE),
        next_batch(LOOKAHEAD_COUNT as usize),
        next_batch(ROWS),
        Some(children_noise_scale),
        prune_noise_scale,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    let batch = [&slot_ancestors, &token_ids, &node_metadata, &ancestor_indices, &valid, &ids]
        .into_iter()
        .flat_map(|buffer| buffer_to_vec::<B, u32>(&context, buffer))
        .chain(buffer_to_vec::<B, f32>(&context, &candidate_logits).into_iter().map(f32::to_bits))
        .collect();
    (buffer_to_vec(&*context, &frontier), buffer_to_vec(&*context, &tree), batch)
}

/// `actual` against `expected` in rows of `capacity` slots: in the slots `expected` marks 1 in row `active_row`, or
/// in every slot without one, the path and edge logprob rows of `logprobs` within their tolerances and the score keys
/// in row `key` that `expected` derives from its path logprobs derived alike from `actual`'s; all else bit for bit.
fn assert_rows_match(
    actual: &[u32],
    expected: &[u32],
    capacity: usize,
    active_row: Option<usize>,
    logprobs: [(usize, f32); 2],
    key: Option<usize>,
) {
    for (offset, (&actual_bits, &expected_bits)) in actual.iter().zip(expected).enumerate() {
        let (row, slot) = (offset / capacity, offset % capacity);
        let path_bits = |values: &[u32]| values[logprobs[0].0 * capacity + slot];
        let active = active_row.is_none_or(|active_row| expected[active_row * capacity + slot] == 1);
        if let Some(&(_, tolerance)) = logprobs.iter().find(|(logprob, _)| active && *logprob == row) {
            assert!((f32::from_bits(actual_bits) - f32::from_bits(expected_bits)).abs() < tolerance);
        } else if active && key == Some(row) && expected_bits == score_key(path_bits(expected)) {
            assert_eq!(actual_bits, score_key(path_bits(actual)));
        } else {
            assert_eq!(actual_bits, expected_bits, "row {row} slot {slot}");
        }
    }
}

fn top_children<B: Backend>(
    children_noise_scale: f32,
    prune_noise_scale: Option<f32>,
) -> (Vec<u32>, Vec<f32>, Option<Vec<f32>>) {
    let (frontier, ..) = run_select::<B>(
        Children,
        children_noise_scale,
        prune_noise_scale,
        &[1; ROWS],
        &[0; TreeIdx::COUNT * TREE_SLOT_COUNT],
    );
    let field = |field: FrontierIdx| {
        frontier[field as usize * FRONTIER_CAPACITY + CHILDREN..(field as usize + 1) * FRONTIER_CAPACITY].to_vec()
    };
    (
        field(FrontierIdx::TokenId),
        field(FrontierIdx::PathLogprobBits).into_iter().map(f32::from_bits).collect(),
        prune_noise_scale.map(|_| field(FrontierIdx::EdgeLogprobBits).into_iter().map(f32::from_bits).collect()),
    )
}

/// Each row's children: its top `CHILDREN` candidates by logit plus `children_noise_scale` times the target sampler's
/// noise at the row's depth seed, ties to the lower token. Scale 0 gives the draft's likeliest tokens, for greedy
/// decoding; scale 1 a sample without replacement that shares the target's noise.
fn reference_children(children_noise_scale: f32) -> Vec<u32> {
    let (residual, logits, ids) = inputs();
    (0..ROWS)
        .flat_map(|row| {
            let seed = DEPTH_SEEDS[DEPTHS[row] as usize];
            let score = |index: usize| {
                logits[index]
                    + residual[index].to_f32()
                    + children_noise_scale * gumbel_float(seed, revidx(ids[index], VOCAB_SIZE))
            };
            let mut indices = (row * CANDIDATES..(row + 1) * CANDIDATES).collect::<Vec<_>>();
            indices.sort_by(|&left, &right| score(right).total_cmp(&score(left)).then(ids[left].cmp(&ids[right])));
            indices.into_iter().take(CHILDREN).map(|index| ids[index]).collect::<Vec<_>>()
        })
        .collect()
}

#[uzu_test]
fn weaver_top_children_matches_cpu() {
    let mut tree = [0u32; TreeIdx::COUNT * TREE_SLOT_COUNT];
    for slot in 0..TREE_SLOT_COUNT {
        tree[TreeIdx::PathLogprobBits as usize * TREE_SLOT_COUNT + slot] = (slot as f32 * -0.25).to_bits();
        tree[TreeIdx::Depth as usize * TREE_SLOT_COUNT + slot] = slot as u32 * 2;
    }
    assert_ne!(reference_children(0.0), reference_children(1.0));
    for (children_noise_scale, prune_noise_scale) in [(0.0, None), (1.0, None), (1.0, Some(PRUNE_NOISE_SCALE))] {
        let (tokens, model, prune) = top_children::<Cpu>(children_noise_scale, prune_noise_scale);
        assert_eq!(tokens, reference_children(children_noise_scale));
        for valid in [[1, 1, 1], [1, 0, 1]] {
            let (expected, ..) = run_select::<Cpu>(Children, children_noise_scale, prune_noise_scale, &valid, &tree);
            let field = |field: FrontierIdx, slot: usize| expected[field as usize * FRONTIER_CAPACITY + slot];
            for slot in 0..FRONTIER_CAPACITY {
                if slot < CHILDREN || valid[slot / CHILDREN - 1] == 0 {
                    assert!((0..FrontierIdx::COUNT).all(|field| expected[field * FRONTIER_CAPACITY + slot] == 42));
                    continue;
                }
                let row = slot / CHILDREN - 1;
                let child = slot % CHILDREN;
                let index = row * CHILDREN + child;
                assert_eq!(field(FrontierIdx::TokenId, slot), tokens[index]);
                assert_eq!(field(FrontierIdx::ParentSlot, slot), (row + 1) as u32);
                assert_eq!(field(FrontierIdx::Depth, slot), (row as u32 + 1) * 2 + 1);
                let path_bits = ((row as f32 + 1.0) * -0.25 + model[index]).to_bits();
                assert_eq!(field(FrontierIdx::PathLogprobBits, slot), path_bits);
                assert_eq!(
                    field(FrontierIdx::EdgeLogprobBits, slot),
                    prune.as_ref().map_or(model[index], |prune| prune[index]).to_bits()
                );
                assert_eq!(field(FrontierIdx::PathScoreKey, slot), score_key(path_bits));
                assert_eq!(field(FrontierIdx::Active, slot), 1);
            }
            for_each_non_cpu_backend!(|B| {
                let (actual, ..) = run_select::<B>(Children, children_noise_scale, prune_noise_scale, &valid, &tree);
                assert_rows_match(
                    &actual,
                    &expected,
                    FRONTIER_CAPACITY,
                    Some(FrontierIdx::Active as usize),
                    [(FrontierIdx::PathLogprobBits as usize, 1e-5), (FrontierIdx::EdgeLogprobBits as usize, 1e-3)],
                    Some(FrontierIdx::PathScoreKey as usize),
                );
            });
        }
    }
}

/// Fused selection, the rows' children and then the next batch's nodes in one 1024-thread block, as Weaver expands.
/// Each CPU winner's path logprob differs from every child of another row as expandable as it by more than twice the
/// tolerance, so rounding cannot change which nodes win. Siblings need no check: the fixture's logits are multiples of
/// 1/8, so siblings are equal or at least 0.125 apart.
#[uzu_test]
fn weaver_fused_selection_matches_cpu() {
    let mut tree = [0u32; TreeIdx::COUNT * TREE_SLOT_COUNT];
    for row in 0..ROWS {
        tree[TreeIdx::PathLogprobBits as usize * TREE_SLOT_COUNT + row + 1] = ((row + 1) as f32 * -0.25).to_bits();
        tree[TreeIdx::Depth as usize * TREE_SLOT_COUNT + row + 1] = DEPTHS[row];
    }
    for children_noise_scale in [0.0, 1.0] {
        for valid in [[1, 1, 1], [1, 0, 1], [0, 0, 1]] {
            let (frontier, expected_tree, batch) = run_select::<Cpu>(Fused, children_noise_scale, None, &valid, &tree);
            let frontier_field = |field: FrontierIdx, slot: usize| frontier[field as usize * FRONTIER_CAPACITY + slot];
            let tree_field = |field: TreeIdx, slot: usize| expected_tree[field as usize * TREE_SLOT_COUNT + slot];
            let children = (CHILDREN..FRONTIER_CAPACITY)
                .filter(|&slot| valid[slot / CHILDREN - 1] != 0)
                .map(|slot| {
                    (
                        frontier_field(FrontierIdx::ParentSlot, slot),
                        frontier_field(FrontierIdx::Depth, slot) < LOOKAHEAD_COUNT,
                        f32::from_bits(frontier_field(FrontierIdx::PathLogprobBits, slot)),
                    )
                })
                .collect::<Vec<_>>();
            for slot in ROWS + 1..TREE_SLOT_COUNT {
                assert_eq!(tree_field(TreeIdx::Valid, slot), 1);
                let parent = tree_field(TreeIdx::ParentSlot, slot);
                let expandable = tree_field(TreeIdx::Depth, slot) < LOOKAHEAD_COUNT;
                let path = f32::from_bits(tree_field(TreeIdx::PathLogprobBits, slot));
                assert!(
                    children
                        .iter()
                        .filter(|child| child.0 != parent && child.1 == expandable)
                        .all(|child| (child.2 - path).abs() > 2.0 * FUSED_TOLERANCE)
                );
            }
            for_each_non_cpu_backend!(|B| {
                let (actual_frontier, actual_tree, actual_batch) =
                    run_select::<B>(Fused, children_noise_scale, None, &valid, &tree);
                let tolerances = |path: usize, edge: usize| [(path, FUSED_TOLERANCE), (edge, FUSED_TOLERANCE)];
                // Fused selection deactivates its winners' frontier slots and keeps their rounded logprobs.
                assert_rows_match(
                    &actual_frontier,
                    &frontier,
                    FRONTIER_CAPACITY,
                    None,
                    tolerances(FrontierIdx::PathLogprobBits as usize, FrontierIdx::EdgeLogprobBits as usize),
                    Some(FrontierIdx::PathScoreKey as usize),
                );
                assert_rows_match(
                    &actual_tree,
                    &expected_tree,
                    TREE_SLOT_COUNT,
                    Some(TreeIdx::Valid as usize),
                    tolerances(TreeIdx::PathLogprobBits as usize, TreeIdx::EdgeLogprobBits as usize),
                    None,
                );
                assert_eq!(actual_batch, batch);
            });
        }
    }
}

/// The prune channel is the log-softmax over the whole pool of logits plus the target sampler's noise at the
/// parent depth's seed, scaled by `prune_noise_scale`; selection and model logprobs stay as without it. The reference
/// takes that noise from `gumbel_float` on purpose, since matching the target sampler is the contract; the seed, the
/// token indexing and the log-sum-exp are computed here independently, in f64.
#[uzu_test]
fn weaver_top_children_prune_logprobs_match_reference() {
    let (residual, candidate_logits, ids) = inputs();
    let (tokens, model, none) = top_children::<Cpu>(1.0, None);
    let (prune_tokens, prune_model, prune) = top_children::<Cpu>(1.0, Some(PRUNE_NOISE_SCALE));
    assert!(none.is_none());
    let prune = prune.unwrap();
    assert_eq!(prune_tokens, tokens);
    assert_eq!(
        prune_model.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        model.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
    );

    for row in 0..ROWS {
        let seed = DEPTH_SEEDS[DEPTHS[row] as usize];
        let values = (0..CANDIDATES)
            .map(|index| {
                let flat = row * CANDIDATES + index;
                candidate_logits[flat] as f64
                    + residual[flat].to_f64()
                    + PRUNE_NOISE_SCALE as f64 * gumbel_float(seed, revidx(ids[flat], VOCAB_SIZE)) as f64
            })
            .collect::<Vec<_>>();
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_sum = values.iter().map(|value| (value - max).exp()).sum::<f64>().ln() + max;
        for child in 0..CHILDREN {
            let token = tokens[row * CHILDREN + child];
            let index = (0..CANDIDATES).find(|&index| ids[row * CANDIDATES + index] == token).unwrap();
            let actual = prune[row * CHILDREN + child];
            // An f32 log-sum-exp over 512 values of magnitude below 20 keeps about 1e-4 of absolute accuracy.
            assert!((actual as f64 - (values[index] - log_sum)).abs() < 1e-4);
            assert!(actual <= 0.0);
        }
    }
    assert!(prune.iter().zip(&model).any(|(prune, model)| (prune - model).abs() > 0.1));
}
