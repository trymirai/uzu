use std::collections::BTreeSet;

use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Encoder, Kernels,
            gpu_types::{
                trie::TrieNode,
                weaver::{DraftSamplingParams, FRONTIER_NO_WINNER, FrontierIdx, MetadataIdx, TreeIdx},
            },
            kernel::{WeaverFrontierInsertChildrenKernel, WeaverTopChildrenKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        sampling::{Sampling, SamplingMethod, gumbel_float, revidx},
        weaver::WeaverDraftSampling,
    },
    tests::helpers::{
        alloc_allocation, alloc_allocation_with_data, allocation_to_vec, create_context, for_each_non_cpu_backend,
    },
};

const VOCAB_SIZE: u32 = 4096;
const PRUNE_NOISE_SCALE: f32 = 1.0 / 1.5;
const NEUTRAL: DraftSamplingParams = DraftSamplingParams {
    recip_temperature: 1.0,
    top_k: u32::MAX,
    top_p: f32::MAX,
    min_p: 0.0,
};

/// One candidate pool repeated over rows; row `r` expands at depth `r` with its own seed, so rows are independent
/// draws of the same node.
struct Pool {
    candidate_logits: Vec<f32>,
    residual_logits: Vec<bf16>,
    token_ids: Vec<u32>,
    rows: usize,
}

impl Pool {
    fn candidates(&self) -> usize {
        self.token_ids.len()
    }

    fn logits(&self) -> Vec<f32> {
        self.candidate_logits
            .iter()
            .zip(&self.residual_logits)
            .map(|(logit, residual)| logit + residual.to_f32())
            .collect()
    }

    fn seed(row: usize) -> u64 {
        0x9E37_79B9_7F4A_7C15u64.wrapping_mul(row as u64 + 1)
    }
}

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*state >> 40) as f32 / (1u64 << 24) as f32
}

// Coarse grids make many equal logits, and ids run against pool order, so ties are broken by token id.
fn tied_pool(
    candidates: usize,
    rows: usize,
) -> Pool {
    let mut state = 7;
    Pool {
        candidate_logits: (0..candidates).map(|_| ((lcg(&mut state) - 0.5) * 64.0).round() * 0.125).collect(),
        residual_logits: (0..candidates)
            .map(|_| bf16::from_f32(((lcg(&mut state) - 0.5) * 12.0).round() * 0.25))
            .collect(),
        token_ids: (0..candidates)
            .map(|index| ((candidates - index) * 2_654_435_761 % VOCAB_SIZE as usize) as u32)
            .collect(),
        rows,
    }
}

// One head and a tail 30 nats down: in f32 the tail adds nothing to the norm, so the head alone has mass 1 and
// top_p = 1 cuts the tail, as it does in the target sampler.
fn far_tail_pool(rows: usize) -> Pool {
    let candidates = 64;
    Pool {
        candidate_logits: (0..candidates)
            .map(|index| {
                if index == 0 {
                    0.0
                } else {
                    -30.0 - index as f32 / 64.0
                }
            })
            .collect(),
        residual_logits: vec![bf16::ZERO; candidates],
        token_ids: (0..candidates).map(|index| (index * 37 + 5) as u32).collect(),
        rows,
    }
}

// Equal -0 and +0 at the top-k boundary: the target orders them by token id, and -0 has the smaller one.
fn signed_zero_pool(rows: usize) -> Pool {
    Pool {
        candidate_logits: vec![1.0, 0.5, 0.0, -0.0, -1.0, -2.0, -3.0, -4.0],
        residual_logits: [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0].map(bf16::from_f32).to_vec(),
        token_ids: vec![700, 600, 500, 400, 300, 200, 100, 50],
        rows,
    }
}

struct Children {
    tokens: Vec<u32>,
    model: Vec<f32>,
    prune: Option<Vec<f32>>,
}

fn top_children<B: Backend>(
    pool: &Pool,
    expand_width: usize,
    prune_noise_scale: Option<f32>,
    draft_sampling: Option<DraftSamplingParams>,
) -> Children {
    let (rows, candidates) = (pool.rows, pool.candidates());
    let context = create_context::<B>();
    let repeat = |values: &[f32]| values.iter().copied().cycle().take(rows * candidates).collect::<Vec<_>>();
    let residual_logits = pool.residual_logits.iter().copied().cycle().take(rows * candidates).collect::<Vec<_>>();
    let token_ids = pool.token_ids.iter().copied().cycle().take(rows * candidates).collect::<Vec<_>>();
    let mut metadata = vec![0u32; rows * MetadataIdx::COUNT];
    for row in 0..rows {
        metadata[MetadataIdx::Depth as usize * rows + row] = row as u32;
    }
    let seeds = (0..rows).map(Pool::seed).collect::<Vec<_>>();
    let residual_logits = alloc_allocation_with_data::<B, bf16>(&context, &residual_logits);
    let candidate_logits = alloc_allocation_with_data::<B, f32>(&context, &repeat(&pool.candidate_logits));
    let token_ids = alloc_allocation_with_data::<B, u32>(&context, &token_ids);
    let seeds = alloc_allocation_with_data::<B, u64>(&context, &seeds);
    let metadata = alloc_allocation_with_data::<B, u32>(&context, &metadata);
    let mut output_tokens = alloc_allocation::<B, u32>(&context, rows * expand_width);
    let mut output_model = alloc_allocation::<B, f32>(&context, rows * expand_width);
    let mut output_prune = prune_noise_scale.map(|_| alloc_allocation::<B, f32>(&context, rows * expand_width));
    let kernel = <B::Kernels as Kernels>::WeaverTopChildrenKernel::new(
        &context,
        prune_noise_scale.is_some(),
        draft_sampling.is_some(),
    )
    .unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &residual_logits,
        &candidate_logits,
        &token_ids,
        &seeds,
        &metadata,
        &mut output_tokens,
        &mut output_model,
        output_prune.as_mut(),
        rows as u32,
        candidates as u32,
        expand_width as u32,
        VOCAB_SIZE,
        prune_noise_scale,
        draft_sampling,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    Children {
        tokens: allocation_to_vec(&output_tokens),
        model: allocation_to_vec(&output_model),
        prune: output_prune.as_ref().map(allocation_to_vec),
    }
}

/// The target's own CPU sampler on the pool laid out over the vocabulary (every other token at -inf), one row per
/// seed.
fn target_samples(
    pool: &Pool,
    sampling_method: &SamplingMethod,
) -> Vec<u32> {
    let context = create_context::<Cpu>();
    let mut logits = vec![f32::NEG_INFINITY; pool.rows * VOCAB_SIZE as usize];
    for row in 0..pool.rows {
        for (logit, token) in pool.logits().into_iter().zip(&pool.token_ids) {
            logits[row * VOCAB_SIZE as usize + *token as usize] = logit;
        }
    }
    let seeds = (0..pool.rows).map(Pool::seed).collect::<Vec<_>>();
    let logits = alloc_allocation_with_data::<Cpu, f32>(&context, &logits);
    let seeds = alloc_allocation_with_data::<Cpu, u64>(&context, &seeds);
    let nodes = (0..pool.rows as u32)
        .map(|index| TrieNode {
            trie_start: index,
            trie_end: pool.rows as u32 - 1,
            height: index,
        })
        .collect::<Box<[_]>>();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    let sampled = Sampling::new(DataType::F32, VOCAB_SIZE)
        .encode(
            &logits,
            Some(&seeds),
            None,
            None,
            None,
            sampling_method,
            &BatchTopology::new(&nodes, true),
            0..pool.rows as u32,
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    sampled.copyout::<u32>().to_vec()
}

fn stochastic(
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
) -> SamplingMethod {
    SamplingMethod::Stochastic {
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty: None,
        suffix_repetition_length: None,
    }
}

/// The target's filters written out from their definition in unified_sampling.metal, in f64: order by logit at the
/// temperature (then smaller token id), and drop a candidate once top_k are above it, or their mass reaches top_p, or
/// its logit falls below the maximum plus ln(min_p).
fn reference_kept(
    scaled_logits: &[f32],
    token_ids: &[u32],
    top_k: Option<u32>,
    top_p: Option<f64>,
    min_p: Option<f64>,
) -> Vec<usize> {
    let mut order = (0..scaled_logits.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        (scaled_logits[right] as f64)
            .partial_cmp(&(scaled_logits[left] as f64))
            .unwrap()
            .then(token_ids[left].cmp(&token_ids[right]))
    });
    let maximum = scaled_logits[order[0]] as f64;
    let norm = scaled_logits.iter().map(|&logit| (logit as f64 - maximum).exp()).sum::<f64>();
    let mut mass = 0.0;
    let mut kept = Vec::new();
    for (rank, &index) in order.iter().enumerate() {
        let logit = scaled_logits[index] as f64;
        if top_k.is_some_and(|top_k| rank >= top_k as usize)
            || top_p.is_some_and(|top_p| mass >= top_p)
            || min_p.is_some_and(|min_p| logit < maximum + min_p.ln())
        {
            break;
        }
        kept.push(index);
        mass += (logit - maximum).exp() / norm;
    }
    kept
}

/// Thresholds halfway between neighbouring prefixes of the reference order, so an f32 kernel and the f64 reference
/// can only disagree through a bug, never through rounding.
fn midpoint_top_p(
    scaled_logits: &[f32],
    token_ids: &[u32],
    rank: usize,
) -> f64 {
    let all = reference_kept(scaled_logits, token_ids, None, None, None);
    let maximum = scaled_logits[all[0]] as f64;
    let norm = scaled_logits.iter().map(|&logit| (logit as f64 - maximum).exp()).sum::<f64>();
    let prefix = |count: usize| {
        all[..count].iter().map(|&index| (scaled_logits[index] as f64 - maximum).exp() / norm).sum::<f64>()
    };
    (prefix(rank) + prefix(rank + 1)) / 2.0
}

fn midpoint_min_p(
    scaled_logits: &[f32],
    token_ids: &[u32],
    rank: usize,
) -> f64 {
    let all = reference_kept(scaled_logits, token_ids, None, None, None);
    let maximum = scaled_logits[all[0]] as f64;
    let below = all[rank..]
        .iter()
        .map(|&index| scaled_logits[index] as f64)
        .find(|&logit| logit < scaled_logits[all[rank - 1]] as f64)
        .unwrap();
    (((scaled_logits[all[rank - 1]] as f64 + below) / 2.0) - maximum).exp()
}

fn scaled(
    pool: &Pool,
    temperature: Option<f32>,
) -> Vec<f32> {
    let recip_temperature = temperature.map_or(1.0, f32::recip);
    pool.logits().into_iter().map(|logit| logit * recip_temperature).collect()
}

/// With the drafter's pool equal to the target's logits, the first child is the target's own sample: both take the
/// Gumbel maximum over the kept set at the target's temperature, with the same noise (`gumbel_float` on CPU). This
/// ties the temperature and the filters in the selection path to the target sampler's code, not to a transcription.
#[uzu_test]
fn draft_sampling_first_child_matches_target_sampler() {
    let methods = [
        stochastic(Some(1.0), Some(20), Some(0.95), None),
        stochastic(Some(0.5), Some(7), None, None),
        stochastic(Some(1.7), None, Some(0.8), None),
        stochastic(Some(0.7), None, None, Some(0.05)),
        stochastic(Some(1.3), Some(50), Some(0.9), Some(0.02)),
        stochastic(Some(0.6), None, None, None),
        stochastic(None, Some(1), None, None),
    ];
    for pool in [tied_pool(512, 96), tied_pool(300, 96)] {
        for method in &methods {
            let draft_sampling = WeaverDraftSampling::from_sampling_method(method).unwrap();
            let children = top_children::<Cpu>(&pool, 1, None, Some(draft_sampling.params()));
            assert_eq!(children.tokens, target_samples(&pool, method), "{method:?}");
        }
    }
    let signed_zero = signed_zero_pool(256);
    let method = stochastic(None, Some(3), None, None);
    let draft_sampling = WeaverDraftSampling::from_sampling_method(&method).unwrap();
    let children = top_children::<Cpu>(&signed_zero, 1, None, Some(draft_sampling.params()));
    let target = target_samples(&signed_zero, &method);
    assert_eq!(children.tokens, target);
    // The boundary token is actually drawn, so a wrong order of -0 and +0 could not pass unseen.
    assert!(target.contains(&400) && !target.contains(&500));
}

/// The whole kept set, read out with one child per candidate, equals the reference, and the children are the kept
/// candidates by descending logit at the temperature plus noise, then the sentinel.
#[uzu_test]
fn draft_sampling_kept_set_matches_reference() {
    let pool = tied_pool(512, 3);
    for temperature in [None, Some(0.5), Some(1.7)] {
        let scaled_logits = scaled(&pool, temperature);
        let top_p = |rank| Some(midpoint_top_p(&scaled_logits, &pool.token_ids, rank));
        let min_p = |rank| Some(midpoint_min_p(&scaled_logits, &pool.token_ids, rank));
        for (top_k, top_p, min_p) in [
            (Some(1), None, None),
            (Some(7), None, None),
            (Some(512), None, None),
            (Some(600), None, None),
            (None, top_p(5), None),
            (None, top_p(40), None),
            (None, None, min_p(12)),
            (Some(30), top_p(20), min_p(25)),
        ] {
            let draft_sampling = WeaverDraftSampling {
                temperature,
                top_k,
                top_p: top_p.map(|top_p| top_p as f32),
                min_p: min_p.map(|min_p| min_p as f32),
            };
            let children = top_children::<Cpu>(&pool, pool.candidates(), None, Some(draft_sampling.params()));
            let kept = reference_kept(&scaled_logits, &pool.token_ids, top_k, top_p, min_p);
            for row in 0..pool.rows {
                let row_tokens = &children.tokens[row * pool.candidates()..(row + 1) * pool.candidates()];
                let mut expected = kept.clone();
                // Children come in the order of the selection key, logit plus the depth's noise, ties by token id.
                expected.sort_by(|&left, &right| {
                    let key = |index: usize| {
                        scaled_logits[index] + gumbel_float(Pool::seed(row), revidx(pool.token_ids[index], VOCAB_SIZE))
                    };
                    key(right).total_cmp(&key(left)).then(pool.token_ids[left].cmp(&pool.token_ids[right]))
                });
                let expected_tokens = expected.iter().map(|&index| pool.token_ids[index]).collect::<Vec<_>>();
                assert_eq!(&row_tokens[..kept.len()], expected_tokens, "{temperature:?} {top_k:?} {top_p:?} {min_p:?}");
                assert!(row_tokens[kept.len()..].iter().all(|&token| token == FRONTIER_NO_WINNER));
            }
        }
    }
}

/// Expansion and pruning weights are log-softmaxes over the kept set only, at the target's temperature:
/// `l' − LSE_S(l')` and `(l' + G / sigma) − LSE_S(l' + G / sigma)`, recomputed here in f64.
#[uzu_test]
fn draft_sampling_normalises_over_the_kept_set() {
    let pool = tied_pool(512, 3);
    for method in [stochastic(Some(0.5), Some(40), Some(0.9), None), stochastic(Some(1.7), None, None, Some(0.01))] {
        let draft_sampling = WeaverDraftSampling::from_sampling_method(&method).unwrap();
        let children =
            top_children::<Cpu>(&pool, pool.candidates(), Some(PRUNE_NOISE_SCALE), Some(draft_sampling.params()));
        let scaled_logits = scaled(&pool, draft_sampling.temperature);
        let index_of = |token: u32| pool.token_ids.iter().position(|&id| id == token).unwrap();
        for row in 0..pool.rows {
            let range = row * pool.candidates()..(row + 1) * pool.candidates();
            let kept = children.tokens[range.clone()]
                .iter()
                .filter(|&&token| token != FRONTIER_NO_WINNER)
                .map(|&token| index_of(token))
                .collect::<Vec<_>>();
            let log_sum = |values: &dyn Fn(usize) -> f64| {
                let maximum = kept.iter().map(|&index| values(index)).fold(f64::NEG_INFINITY, f64::max);
                kept.iter().map(|&index| (values(index) - maximum).exp()).sum::<f64>().ln() + maximum
            };
            let model = |index: usize| scaled_logits[index] as f64;
            let prune = |index: usize| {
                scaled_logits[index] as f64
                    + PRUNE_NOISE_SCALE as f64
                        * gumbel_float(Pool::seed(row), revidx(pool.token_ids[index], VOCAB_SIZE)) as f64
            };
            let (model_log_sum, prune_log_sum) = (log_sum(&model), log_sum(&prune));
            let prune_output = &children.prune.as_ref().unwrap()[range.clone()];
            for (child, &index) in kept.iter().enumerate() {
                // An f32 log-sum-exp over 512 values of magnitude below 40 keeps about 1e-4 of absolute accuracy.
                assert!((children.model[range.start + child] as f64 - (model(index) - model_log_sum)).abs() < 1e-4);
                assert!((prune_output[child] as f64 - (prune(index) - prune_log_sum)).abs() < 1e-4);
            }
            let total = |values: &[f32]| values[..kept.len()].iter().map(|&value| (value as f64).exp()).sum::<f64>();
            assert!((total(&children.model[range.clone()]) - 1.0).abs() < 1e-4);
            assert!((total(prune_output) - 1.0).abs() < 1e-4);
        }
    }
}

/// Parameters that never bind leave the kernel's outputs bit for bit as without draft sampling, far tail included,
/// where a top_p of 1 would already drop candidates.
#[uzu_test]
fn draft_sampling_neutral_parameters_are_bit_exact() {
    for pool in [tied_pool(512, 3), tied_pool(300, 3), far_tail_pool(3)] {
        for prune_noise_scale in [None, Some(PRUNE_NOISE_SCALE)] {
            let off = top_children::<Cpu>(&pool, 8, prune_noise_scale, None);
            for params in [
                NEUTRAL,
                DraftSamplingParams {
                    top_k: pool.candidates() as u32,
                    ..NEUTRAL
                },
            ] {
                let on = top_children::<Cpu>(&pool, 8, prune_noise_scale, Some(params));
                assert_eq!(on.tokens, off.tokens);
                let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&on.model), bits(&off.model));
                assert_eq!(on.prune.as_deref().map(bits), off.prune.as_deref().map(bits));
            }
        }
    }
    // The same tail is cut once top_p is 1: its mass above rounds to 1 in f32, as in the target sampler.
    let pool = far_tail_pool(1);
    let cut = top_children::<Cpu>(
        &pool,
        pool.candidates(),
        None,
        Some(DraftSamplingParams {
            top_p: 1.0,
            ..NEUTRAL
        }),
    );
    assert!(cut.tokens.contains(&FRONTIER_NO_WINNER));
}

/// When fewer candidates survive than there are children, the rest are sentinels with no weight in either channel.
#[uzu_test]
fn draft_sampling_marks_missing_children() {
    let pool = tied_pool(512, 3);
    let params = DraftSamplingParams {
        top_k: 3,
        ..NEUTRAL
    };
    let children = top_children::<Cpu>(&pool, 8, Some(PRUNE_NOISE_SCALE), Some(params));
    let kept = reference_kept(&pool.logits(), &pool.token_ids, Some(3), None, None)
        .into_iter()
        .map(|index| pool.token_ids[index])
        .collect::<BTreeSet<_>>();
    let prune = children.prune.unwrap();
    for row in 0..pool.rows {
        let range = row * 8..(row + 1) * 8;
        assert_eq!(children.tokens[range.start..range.start + 3].iter().copied().collect::<BTreeSet<_>>(), kept);
        for child in range.start + 3..range.end {
            assert_eq!(children.tokens[child], FRONTIER_NO_WINNER);
            assert_eq!(children.model[child], f32::NEG_INFINITY);
            assert_eq!(prune[child], f32::NEG_INFINITY);
        }
    }
}

fn insert_children<B: Backend>(child_ids: &[u32]) -> Vec<u32> {
    let (slots, nodes, width, capacity) = (4usize, 2usize, 3usize, 12usize);
    let context = create_context::<B>();
    let mut tree = vec![0u32; TreeIdx::COUNT * slots];
    tree[TreeIdx::PathLogprobBits as usize * slots + 1] = (-0.5f32).to_bits();
    let mut metadata = vec![0u32; MetadataIdx::COUNT * nodes];
    metadata[MetadataIdx::TreeSlot as usize * nodes + 1] = 1;
    let tree = alloc_allocation_with_data::<B, u32>(&context, &tree);
    let metadata = alloc_allocation_with_data::<B, u32>(&context, &metadata);
    let valid = alloc_allocation_with_data::<B, u32>(&context, &[1, 1]);
    let child_ids = alloc_allocation_with_data::<B, u32>(&context, child_ids);
    let logprobs = alloc_allocation_with_data::<B, f32>(&context, &[-0.25f32; 6]);
    let mut frontier = alloc_allocation_with_data::<B, u32>(&context, &vec![42u32; FrontierIdx::COUNT * capacity]);
    let kernel = <B::Kernels as Kernels>::WeaverFrontierInsertChildrenKernel::new(&context).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &tree,
        &metadata,
        &valid,
        &child_ids,
        &logprobs,
        &logprobs,
        &mut frontier,
        capacity as u32,
        slots as u32,
        nodes as u32,
        width as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&frontier)
}

/// A sentinel child leaves its frontier slot untouched, and its siblings are inserted as usual.
#[uzu_test]
fn insert_children_skips_missing_children() {
    let with_sentinel = insert_children::<Cpu>(&[10, 11, 12, 20, FRONTIER_NO_WINNER, 22]);
    let without = insert_children::<Cpu>(&[10, 11, 12, 20, 21, 22]);
    let capacity = with_sentinel.len() / FrontierIdx::COUNT;
    let field = |frontier: &[u32], field: FrontierIdx, slot: usize| frontier[field as usize * capacity + slot];
    for slot in 0..capacity {
        if slot == 4 {
            for index in 0..FrontierIdx::COUNT {
                assert_eq!(with_sentinel[index * capacity + slot], 42);
            }
        } else {
            for index in 0..FrontierIdx::COUNT {
                assert_eq!(with_sentinel[index * capacity + slot], without[index * capacity + slot]);
            }
        }
    }
    assert_eq!(field(&without, FrontierIdx::TokenId, 4), 21);
    assert_eq!(field(&without, FrontierIdx::Active, 4), 1);
}

/// Metal against the CPU twin: the kept sets agree as sets and the first children in order (the selection noise
/// differs in its last bits between the two, so the full order may flip at near ties), the weights agree to fast-math
/// accuracy, and neutral parameters are bit-exact on Metal as well.
#[uzu_test]
fn draft_sampling_matches_cpu() {
    let methods = [
        stochastic(Some(1.0), Some(20), Some(0.95), None),
        stochastic(Some(0.5), Some(7), None, Some(0.05)),
        stochastic(Some(1.7), None, Some(0.8), None),
    ];
    for pool in [tied_pool(512, 4), tied_pool(300, 4)] {
        for method in &methods {
            let params = WeaverDraftSampling::from_sampling_method(method).unwrap().params();
            for_each_non_cpu_backend!(|B| {
                let expected = top_children::<Cpu>(&pool, pool.candidates(), Some(PRUNE_NOISE_SCALE), Some(params));
                let actual = top_children::<B>(&pool, pool.candidates(), Some(PRUNE_NOISE_SCALE), Some(params));
                for row in 0..pool.rows {
                    let range = row * pool.candidates()..(row + 1) * pool.candidates();
                    let set = |tokens: &[u32]| tokens.iter().copied().collect::<BTreeSet<_>>();
                    assert_eq!(set(&actual.tokens[range.clone()]), set(&expected.tokens[range.clone()]));
                    assert_eq!(
                        actual.tokens[range.start..range.start + 8],
                        expected.tokens[range.start..range.start + 8]
                    );
                    for child in range.start..range.start + 8 {
                        // Missing children carry -inf on both backends; their difference would be NaN.
                        if expected.tokens[child] == FRONTIER_NO_WINNER {
                            assert_eq!(actual.model[child], f32::NEG_INFINITY);
                        } else {
                            assert!((actual.model[child] - expected.model[child]).abs() < 1e-3);
                        }
                    }
                }
            });
        }
        for_each_non_cpu_backend!(|B| {
            let off = top_children::<B>(&pool, 8, Some(PRUNE_NOISE_SCALE), None);
            let on = top_children::<B>(&pool, 8, Some(PRUNE_NOISE_SCALE), Some(NEUTRAL));
            assert_eq!(on.tokens, off.tokens);
            let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&on.model), bits(&off.model));
            assert_eq!(on.prune.as_deref().map(bits), off.prune.as_deref().map(bits));
        });
    }
    let sentinel = [10, 11, 12, 20, FRONTIER_NO_WINNER, 22];
    for_each_non_cpu_backend!(|B| {
        assert_eq!(insert_children::<B>(&sentinel), insert_children::<Cpu>(&sentinel));
    });
}
