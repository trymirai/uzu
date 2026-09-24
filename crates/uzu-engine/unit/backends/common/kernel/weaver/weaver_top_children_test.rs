use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Encoder, Kernels, gpu_types::weaver::MetadataIdx, kernel::WeaverTopChildrenKernel},
        cpu::Cpu,
    },
    encodable_block::sampling::{gumbel_float, revidx},
    tests::helpers::{
        alloc_allocation, alloc_allocation_with_data, allocation_to_vec, create_context, for_each_non_cpu_backend,
    },
};

const CANDIDATES: usize = 512;
const CHILDREN: usize = 8;
const VOCAB_SIZE: u32 = 131_072;
const DEPTHS: [u32; 3] = [0, 1, 2];
const DEPTH_SEEDS: [u64; 3] = [0x9E3779B97F4A7C15, 0xD1B54A32D192ED03, 0x2545F4914F6CDD1D];
const ROWS: usize = 3;
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

fn top_children<B: Backend>(
    residual: &[bf16],
    candidate_logits: &[f32],
    ids: &[u32],
    prune_noise_scale: Option<f32>,
) -> (Vec<u32>, Vec<f32>, Option<Vec<f32>>) {
    let rows = residual.len() / CANDIDATES;
    assert_eq!(residual.len(), rows * CANDIDATES);
    assert_eq!(candidate_logits.len(), rows * CANDIDATES);
    assert_eq!(ids.len(), rows * CANDIDATES);
    assert!(rows <= DEPTHS.len());
    let context = create_context::<B>();
    let residual = alloc_allocation_with_data::<B, bf16>(&context, residual);
    let candidate_logits = alloc_allocation_with_data::<B, f32>(&context, candidate_logits);
    let ids = alloc_allocation_with_data::<B, u32>(&context, ids);
    let depth_seeds = alloc_allocation_with_data::<B, u64>(&context, &DEPTH_SEEDS);
    let mut metadata_values = vec![0u32; rows * MetadataIdx::COUNT];
    for row in 0..rows {
        metadata_values[MetadataIdx::Depth as usize * rows + row] = DEPTHS[row];
    }
    let node_metadata = alloc_allocation_with_data::<B, u32>(&context, &metadata_values);
    let mut output_token_ids = alloc_allocation::<B, u32>(&context, rows * CHILDREN);
    let mut output_model_logprobs = alloc_allocation::<B, f32>(&context, rows * CHILDREN);
    let mut output_prune_logprobs = prune_noise_scale.map(|_| alloc_allocation::<B, f32>(&context, rows * CHILDREN));
    let kernel = <B::Kernels as Kernels>::WeaverTopChildrenKernel::new(&context, prune_noise_scale.is_some()).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &residual,
        &candidate_logits,
        &ids,
        &depth_seeds,
        &node_metadata,
        &mut output_token_ids,
        &mut output_model_logprobs,
        output_prune_logprobs.as_mut(),
        rows as u32,
        CANDIDATES as u32,
        CHILDREN as u32,
        VOCAB_SIZE,
        prune_noise_scale,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    (
        allocation_to_vec(&output_token_ids),
        allocation_to_vec(&output_model_logprobs),
        output_prune_logprobs.as_ref().map(allocation_to_vec),
    )
}

#[uzu_test]
fn weaver_top_children_matches_cpu() {
    let (residual, candidate_logits, ids) = inputs();

    for prune_noise_scale in [None, Some(PRUNE_NOISE_SCALE)] {
        for_each_non_cpu_backend!(|B| {
            let expected = top_children::<Cpu>(&residual, &candidate_logits, &ids, prune_noise_scale);
            let actual = top_children::<B>(&residual, &candidate_logits, &ids, prune_noise_scale);
            assert_eq!(actual.0, expected.0);
            for (actual, expected) in actual.1.iter().zip(expected.1) {
                assert!((actual - expected).abs() < 1e-5);
            }
            assert_eq!(actual.2.is_some(), expected.2.is_some());
            // The prune channel also carries the Gumbel transform, which Metal evaluates with fast-math logs.
            for (actual, expected) in actual.2.iter().flatten().zip(expected.2.iter().flatten()) {
                assert!((actual - expected).abs() < 1e-3);
            }
        });
    }
}

/// The prune channel is the log-softmax over the whole pool of logits plus the target sampler's noise at the
/// parent depth's seed, scaled by `prune_noise_scale`; selection and model logprobs stay as without it. The reference
/// takes that noise from `gumbel_float` on purpose, since matching the target sampler is the contract; the seed, the
/// token indexing and the log-sum-exp are computed here independently, in f64.
#[uzu_test]
fn weaver_top_children_prune_logprobs_match_reference() {
    let (residual, candidate_logits, ids) = inputs();
    let (tokens, model, none) = top_children::<Cpu>(&residual, &candidate_logits, &ids, None);
    let (prune_tokens, prune_model, prune) =
        top_children::<Cpu>(&residual, &candidate_logits, &ids, Some(PRUNE_NOISE_SCALE));
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
