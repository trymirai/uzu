use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx, WeaverSelectionMode::Children},
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
const TREE_SLOT_COUNT: usize = ROWS + 1;
const FRONTIER_CAPACITY: usize = TREE_SLOT_COUNT * CHILDREN;
const PRUNE_NOISE_SCALE: f32 = 1.0 / 1.5;

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

fn run_top_children<B: Backend>(
    prune_noise_scale: Option<f32>,
    node_valid: &[u32; ROWS],
    tree_values: &[u32; TreeIdx::COUNT * TREE_SLOT_COUNT],
) -> Vec<u32> {
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
    let mut frontier = create_buffer_with_data::<B, u32>(&context, &vec![42; FrontierIdx::COUNT * FRONTIER_CAPACITY]);
    let kernel =
        <B::Kernels as Kernels>::WeaverSelectKernel::new(&context, 256, Children, prune_noise_scale.is_some()).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        Some(&residual),
        Some(&depth_seeds),
        &mut frontier,
        &mut tree,
        None::<&mut B::ScratchBuffer>,
        None::<&mut B::ScratchBuffer>,
        &mut node_metadata,
        None::<&mut B::ScratchBuffer>,
        &mut valid,
        &pool_ids,
        &pool_logits,
        &mut ids,
        &mut candidate_logits,
        ROWS as u32,
        1,
        CANDIDATES as u32,
        CHILDREN as u32,
        VOCAB_SIZE,
        FRONTIER_CAPACITY as u32,
        TREE_SLOT_COUNT as u32,
        0,
        0,
        0,
        0,
        0,
        prune_noise_scale,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec(&*context, &frontier)
}

fn top_children<B: Backend>(prune_noise_scale: Option<f32>) -> (Vec<u32>, Vec<f32>, Option<Vec<f32>>) {
    let frontier = run_top_children::<B>(prune_noise_scale, &[1; ROWS], &[0; TreeIdx::COUNT * TREE_SLOT_COUNT]);
    let field = |field: FrontierIdx| {
        frontier[field as usize * FRONTIER_CAPACITY + CHILDREN..(field as usize + 1) * FRONTIER_CAPACITY].to_vec()
    };
    (
        field(FrontierIdx::TokenId),
        field(FrontierIdx::PathLogprobBits).into_iter().map(f32::from_bits).collect(),
        prune_noise_scale.map(|_| field(FrontierIdx::EdgeLogprobBits).into_iter().map(f32::from_bits).collect()),
    )
}

#[uzu_test]
fn weaver_top_children_matches_cpu() {
    let mut tree = [0u32; TreeIdx::COUNT * TREE_SLOT_COUNT];
    for slot in 0..TREE_SLOT_COUNT {
        tree[TreeIdx::PathLogprobBits as usize * TREE_SLOT_COUNT + slot] = (slot as f32 * -0.25).to_bits();
        tree[TreeIdx::Depth as usize * TREE_SLOT_COUNT + slot] = slot as u32 * 2;
    }
    for prune_noise_scale in [None, Some(PRUNE_NOISE_SCALE)] {
        let (tokens, model, prune) = top_children::<Cpu>(prune_noise_scale);
        for valid in [[1, 1, 1], [1, 0, 1]] {
            let expected = run_top_children::<Cpu>(prune_noise_scale, &valid, &tree);
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
                let actual = run_top_children::<B>(prune_noise_scale, &valid, &tree);
                for offset in 0..expected.len() {
                    let field_index = offset / FRONTIER_CAPACITY;
                    let slot = offset % FRONTIER_CAPACITY;
                    let active = field(FrontierIdx::Active, slot) == 1;
                    if active
                        && (field_index == FrontierIdx::PathLogprobBits as usize
                            || field_index == FrontierIdx::EdgeLogprobBits as usize)
                    {
                        let tolerance = if field_index == FrontierIdx::PathLogprobBits as usize {
                            1e-5
                        } else {
                            1e-3
                        };
                        assert!((f32::from_bits(actual[offset]) - f32::from_bits(expected[offset])).abs() < tolerance);
                    } else if active && field_index == FrontierIdx::PathScoreKey as usize {
                        let path_bits = actual[FrontierIdx::PathLogprobBits as usize * FRONTIER_CAPACITY + slot];
                        assert_eq!(actual[offset], score_key(path_bits));
                    } else {
                        assert_eq!(actual[offset], expected[offset]);
                    }
                }
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
    let (tokens, model, none) = top_children::<Cpu>(None);
    let (prune_tokens, prune_model, prune) = top_children::<Cpu>(Some(PRUNE_NOISE_SCALE));
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
