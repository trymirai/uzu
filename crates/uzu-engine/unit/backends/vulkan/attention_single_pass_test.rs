use std::{
    collections::BTreeMap,
    fmt::Debug,
    panic::{AssertUnwindSafe, catch_unwind},
    time::{Duration, Instant},
};

use bytemuck::{AnyBitPattern, NoUninit};
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{AttentionSinglePassCase as Case, kernel_fixture::KernelFixture};
use crate::{
    array::ArrayElement,
    backends::{
        common::gpu_types::ring::RingParams,
        vulkan::{AttentionSinglePassVulkanKernel, Error},
    },
    data_type::DataType,
};

/// CPU and Vulkan against the FP64 oracle for every case; prints per-label maxima and fails on any violation.
fn check<T: ArrayElement + Float + Debug + NoUninit + AnyBitPattern>(
    fixture: &KernelFixture,
    group: &str,
    cases: &[Case],
) {
    let mut totals = BTreeMap::<String, [f64; 4]>::new();
    for case in cases {
        let expected = case.oracle::<T>();
        let label = format!("{group} {:?} {}", T::data_type(), case.label());
        let (cpu, vulkan) = (case.cpu::<T>(), case.gpu::<T>(fixture, &case.vulkan_kernel::<T>(fixture)));
        for (backend, actual, reference) in [("CPU", &cpu, None), ("Vulkan", &vulkan, Some(&cpu[..]))] {
            let errors = Case::compare(&expected, actual, reference, &format!("{label} {backend}"));
            let total = totals.entry(format!("{group} {:?} {backend}", T::data_type())).or_default();
            *total = [total[0].max(errors[0]), total[1].max(errors[1]), total[2] + errors[2], total[3] + errors[3]];
        }
    }
    for (label, [absolute, ratio, nans, violations]) in &totals {
        eprintln!(
            "AttentionSinglePass {label}: max error {absolute:.3e}, max error/bound {ratio:.3e}, {nans} NaN, \
             {violations} violations"
        );
    }
    assert!(totals.values().all(|total| total[3] == 0.0), "AttentionSinglePass {group}: elements exceed the bound");
}

fn ring(
    ring_offset: u32,
    ring_length: u32,
) -> Option<RingParams> {
    Some(RingParams {
        ring_offset,
        ring_length,
    })
}

fn check_both(
    group: &str,
    cases: &[Case],
) {
    let fixture = KernelFixture::new();
    check::<f32>(&fixture, group, cases);
    check::<bf16>(&fixture, group, cases);
    fixture.assert_clean();
}

/// Every specialization: sinks, a partially filled wrapped ring, causality, a trie and a window.
#[uzu_test]
fn all_specializations() {
    let cases = (0..32u32)
        .map(|flags| {
            let mut case = Case::new(64, 4, 2, 5, 6, flags);
            case.sinks = (flags & 1 != 0).then(|| vec![0.5, -1.0, 2.0, -3.0]);
            if flags & 2 != 0 {
                case.ring = ring(3, 4);
            }
            case.is_causal = flags & 4 != 0;
            case.window = (flags & 16 != 0).then_some(5);
            if flags & 8 != 0 {
                case = case.with_random_trie(flags);
            }
            case
        })
        .collect::<Vec<_>>();
    check_both("specializations", &cases);
}

/// Every head dimension with GQA 1, 4 and 8 over padded, independent K and V strides, across tile tails.
#[uzu_test]
fn head_dims_gqa_and_strides() {
    let cases = itertools::iproduct!([64, 128, 256, 512], [1, 4, 8])
        .map(|(head_dim, gqa_factor)| Case::new(head_dim, 8, gqa_factor, 127, 3, head_dim + gqa_factor))
        .collect::<Vec<_>>();
    check_both("shapes", &cases);
}

/// The model's contiguous cache layout `[rows, kv_heads, head_dim]`: a decode and a causal prefill.
#[uzu_test]
fn model_layout() {
    let cases = [Case::new(128, 8, 4, 1023, 1, 51), Case::new(64, 4, 2, 100, 30, 52), Case::new(512, 4, 4, 70, 3, 53)]
        .map(Case::with_model_layout);
    check_both("model layout", &cases);
}

/// Ring offsets at the start, middle and end, full and partial, across two tiles, and an empty ring.
#[uzu_test]
fn ring_offsets() {
    let mut cases = itertools::iproduct!([0, 33, 69], [70, 50])
        .map(|(ring_offset, ring_length)| {
            let mut case = Case::new(64, 2, 1, 70, 4, ring_offset + ring_length);
            case.ring = ring(ring_offset, ring_length);
            case
        })
        .collect::<Vec<_>>();
    let mut empty = Case::new(64, 2, 1, 0, 4, 1);
    empty.ring = ring(0, 0);
    cases.push(empty);
    check_both("rings", &cases);
}

/// Tries with and without causality, after a prefix, and under a window.
#[uzu_test]
fn trie_masks() {
    let cases = [(0, true, None), (9, true, None), (9, false, None), (9, true, Some(3)), (70, false, Some(4))]
        .into_iter()
        .map(|(prefix, is_causal, window)| {
            let mut case = Case::new(128, 2, 2, prefix, 40, prefix + 7).with_random_trie(prefix + 1);
            (case.is_causal, case.window) = (is_causal, window);
            case
        })
        .collect::<Vec<_>>();
    check_both("tries", &cases);
}

/// Causal and centered windows of 0 (nothing visible: NaN, or 0 with sinks), 1, 2, odd and beyond the sequence.
#[uzu_test]
fn windows() {
    let mut cases = itertools::iproduct!([0, 1, 2, 5, 1000], [true, false])
        .map(|(window, is_causal)| {
            let mut case = Case::new(64, 2, 1, 20, 8, window);
            (case.window, case.is_causal) = (Some(window), is_causal);
            case
        })
        .collect::<Vec<_>>();
    let mut sunk = cases[0].clone();
    sunk.sinks = Some(vec![0.25, -0.75]);
    cases.push(sunk);
    check_both("windows", &cases);
}

/// Dominant, comparable and negligible sinks.
#[uzu_test]
fn sinks() {
    let cases = [30.0, 0.0, -30.0]
        .map(|sink| {
            let mut case = Case::new(128, 4, 4, 33, 5, 3);
            case.sinks = Some(vec![sink, sink + 1.0, -sink, 0.5]);
            case
        })
        .to_vec();
    check_both("sinks", &cases);
}

/// NaN and infinities in used inputs produce the CPU's classes; in masked rows and unfilled ring slots they are never
/// read. A first used key scoring -inf without a finite sink is NaN; nonfinite sinks; zero scale; signed zeros.
#[uzu_test]
fn nonfinite_inputs() {
    let base = Case::new(64, 2, 1, 6, 5, 11);
    let poisoned = |rows: &[u32], key: f32, value: f32| {
        let mut case = base.clone();
        for (&row, j) in itertools::iproduct!(rows, [0, 7]) {
            let (k, v) = (case.index(case.k_strides, 0, row, j), case.index(case.v_strides, 0, row, j));
            (case.keys[k], case.values[v]) = (key, value);
        }
        case
    };
    let mut cases = vec![
        poisoned(&[2], f32::NAN, 0.5),
        poisoned(&[2], f32::INFINITY, 0.5),
        poisoned(&[3], 0.5, f32::NAN),
        poisoned(&[3], 0.5, f32::INFINITY),
        poisoned(&[1, 3], 0.5, f32::NEG_INFINITY),
        // The last suffix row is visible only to the last query.
        poisoned(&[10], f32::NAN, f32::INFINITY),
    ];
    // Positions 0..4 from slot 1 fill slots 1..=4, leaving the poisoned slots 0 and 5 unfilled.
    let mut unfilled = poisoned(&[0, 5], f32::NAN, f32::NAN);
    unfilled.ring = ring(1, 4);
    cases.push(unfilled);
    for sink in [None, Some(0.5), Some(f32::NEG_INFINITY), Some(f32::NAN), Some(f32::INFINITY)] {
        let mut case = poisoned(&[0], f32::NEG_INFINITY, 0.5);
        case.queries.iter_mut().for_each(|query| *query = query.abs() + 0.125);
        case.sinks = sink.map(|sink| vec![sink; 2]);
        cases.push(case);
    }
    // Nothing visible against a nonfinite sink is 0.
    for sink in [f32::INFINITY, f32::NAN] {
        let mut case = base.clone();
        (case.window, case.sinks) = (Some(0), Some(vec![sink; 2]));
        cases.push(case);
    }
    let mut nan_query = base.clone();
    nan_query.queries[5] = f32::NAN;
    let mut zero_scale = base.clone();
    zero_scale.scale = 0.0;
    let mut zeros = base.clone();
    zeros.values.iter_mut().filter(|value| !value.is_nan()).for_each(|value| *value = -0.0);
    cases.extend([nan_query, zero_scale, zeros.clone()]);
    check_both("nonfinite", &cases);
    // Zeros of either sign sum to +0 on both backends.
    let fixture = KernelFixture::new();
    let kernel = zeros.vulkan_kernel::<f32>(&fixture);
    for output in [zeros.cpu::<f32>(), zeros.gpu::<f32>(&fixture, &kernel)] {
        assert!(output.iter().all(|value| value.to_bits() == 0), "signed zeros: {output:?}");
    }
    fixture.assert_clean();
}

/// Subnormal operands, weights and rescales that the device would flush while they still weigh in normal outputs:
/// subnormal keys under huge queries, a subnormal scaled query under huge keys, a subnormal weight and a subnormal
/// rescale on a huge value, and subnormal values.
#[uzu_test]
fn tiny_values() {
    let base = Case::new(64, 2, 1, 6, 3, 21);
    let scaled = |query: i32, key: i32, value: i32, scale: f32| {
        let mut case = base.clone();
        case.queries.iter_mut().for_each(|x| *x *= 2f32.powi(query));
        case.keys.iter_mut().for_each(|x| *x *= 2f32.powi(key));
        case.values.iter_mut().for_each(|x| *x *= 2f32.powi(value));
        case.scale = scale;
        case
    };
    // One query of ones (scale 1/8) over constant key rows scoring 8 c each; row 0 holds huge values.
    let rows = |prefix: u32, constants: &dyn Fn(u32) -> f32| {
        let mut case = Case::new(64, 1, 1, prefix, 1, 5);
        case.queries.fill(1.0);
        for (row, j) in itertools::iproduct!(0..prefix + 1, 0..64) {
            let index = case.index(case.k_strides, 0, row, j);
            case.keys[index] = constants(row);
            if row == 0 {
                let index = case.index(case.v_strides, 0, row, j);
                case.values[index] = 2f32.powi(126);
            }
        }
        case
    };
    let cases = [
        scaled(124, -125, 0, 1.0),
        scaled(0, 127, 0, 2f32.powi(-130)),
        rows(1, &|row| {
            if row == 0 {
                -95.0 / 8.0
            } else {
                0.0
            }
        }),
        rows(64, &|row| match row {
            0 => 0.0,
            64 => 95.0 / 8.0,
            _ => -25.0,
        }),
        scaled(0, 0, -132, 0.125),
    ];
    check_both("tiny", &cases);
}

/// The last of `rows` rows queries every row with D 64: each `(row, score, value)` key scores exactly `score` with all
/// values `value`; other rows score -inf with zero values. Rows 0 and 64 lie in separate tiles.
fn witness(
    keys: &[(u32, f32, f32)],
    rows: u32,
    sink: Option<f32>,
) -> Case {
    let mut case = Case::new(64, 1, 1, rows - 1, 1, 3);
    (case.scale, case.sinks) = (1.0, sink.map(|sink| vec![sink]));
    case.queries = (0..64)
        .map(|j| {
            if j == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    for (row, j) in itertools::iproduct!(0..rows, 0..64) {
        let (score, value) =
            keys.iter().find(|key| key.0 == row).map_or((f32::NEG_INFINITY, 0.0), |&(_, score, value)| (score, value));
        let (k, v) = (case.index(case.k_strides, 0, row, j), case.index(case.v_strides, 0, row, j));
        case.keys[k] = if j == 0 {
            score
        } else {
            0.0
        };
        case.values[v] = value;
    }
    case
}

/// The CPU's staging where the device would flush, compared with the CPU directly: weights rounded to FP32 onto the
/// subnormal grid before they multiply huge values (e^-90 alone is subnormal, so flushing it loses a normal output),
/// 2^120 on the tile-parallel path and the largest finite value past the magnitude limit on the per-key path, in
/// either key order and across tiles, where the rescale is rounded the same way; an infinity at a weight rounding
/// to 0 is NaN. Both kernels round e^x onto the grid from approximations within u (6 + 9|x|) of it (`finite_bound`),
/// so their weights differ by that much, plus one grid step only where a rounding midpoint lies within it; times the
/// value, plus the quotient (2.5 ULPs against the CPU's 0.5, 3 ULPs of the output) and for BF16 one storage step.
/// This misses an unrounded weight at -100 and -103.5 by far. Single subnormal and zero values at weight exactly 1
/// stay bit for bit, -0 summing to +0 as on the CPU.
fn staged<T: ArrayElement + Float + Debug + NoUninit + AnyBitPattern>(fixture: &KernelFixture) {
    let huge = T::max_value().to_f32().unwrap();
    let mut cases = Vec::new();
    for (score, value, gap, swapped) in itertools::iproduct!(
        [-90.0f32, -100.0, -103.5, -104.0, -200.0],
        [2f32.powi(120), huge, f32::INFINITY],
        [0, 63],
        [false, true]
    ) {
        let (grid, weight) = (2f64.powi(-149), f64::from(score).exp());
        let tolerance = 2f64.powi(-24) * (6.0 + 9.0 * f64::from(-score)) * weight;
        let midpoint = ((weight / grid).fract() - 0.5).abs() * grid <= tolerance;
        let step = if midpoint {
            grid
        } else {
            0.0
        };
        let slack = (tolerance + step) * f64::from(value);
        let (tiny, top) = ((score, value), (0.0, 0.0));
        let (first, last) = if swapped {
            (top, tiny)
        } else {
            (tiny, top)
        };
        cases.push((witness(&[(0, first.0, first.1), (gap + 1, last.0, last.1)], gap + 2, None), slack));
    }
    for sink in [None, Some(0.5), Some(f32::NEG_INFINITY), Some(f32::NAN), Some(f32::INFINITY)] {
        cases.push((witness(&[(0, f32::NEG_INFINITY, 0.5), (1, 0.0, 0.25)], 2, sink), 0.0));
        cases.push((witness(&[(0, 0.0, 0.25), (64, f32::NEG_INFINITY, 0.5)], 65, sink), 0.0));
    }
    // The CPU's per-key order decides these classes: an infinity whose weight shrinks in steps that each stay nonzero
    // survives, whether the steps fall within a tile or across tiles; descending order rounds its weight to 0 at once
    // (NaN); a finite sum overflowing to infinity keeps it through a nonzero rescale and turns NaN at a zero one.
    for infinity in [f32::INFINITY, f32::NEG_INFINITY] {
        for (keys, rows) in [
            (vec![(0, -200.0, infinity), (1, -100.0, 0.0), (2, 0.0, 0.0)], 3),
            (vec![(0, -200.0, infinity), (64, -100.0, 0.0), (65, 0.0, 0.0)], 66),
            (vec![(0, -200.0, infinity), (1, -100.0, 0.0), (64, 0.0, 0.0)], 65),
            (vec![(0, 0.0, 0.0), (1, -100.0, 0.0), (2, -200.0, infinity)], 3),
        ] {
            cases.push((witness(&keys, rows, None), 0.0));
        }
        for sink in [f32::NEG_INFINITY, 0.5] {
            cases.push((witness(&[(0, -200.0, infinity), (1, -100.0, 0.0), (2, 0.0, 0.0)], 3, Some(sink)), 0.0));
        }
    }
    for (keys, rows) in [
        (vec![(0, 0.0, huge), (1, 0.0, huge), (2, 1.0, 0.0)], 3),
        (vec![(0, 0.0, huge), (1, 0.0, huge), (2, 200.0, 0.0)], 3),
        (vec![(0, 0.0, huge), (64, 0.0, huge), (65, 1.0, 0.0)], 66),
    ] {
        cases.push((witness(&keys, rows, None), 0.0));
    }
    // 65 keys scoring 0: in key order 2^103 - 2^103 + (2^103 - 2^79) + 2^79 is exactly 2^103, and adding f32::MAX ties
    // halfway to 2^128, rounding to infinity; summed by slices first, 2^79 is rounded away beside 2^103 and the sum
    // stays finite. FP32 storage only.
    if size_of::<T>() == 4 {
        let special = [
            (0, 2f32.powi(103)),
            (1, -2f32.powi(103)),
            (2, f32::from_bits(0x72ff_ffff)),
            (16, 2f32.powi(79)),
            (64, f32::MAX),
        ];
        let keys = (0..65)
            .map(|row| (row, 0.0, special.iter().find(|key| key.0 == row).map_or(0.0, |key| key.1)))
            .collect::<Vec<_>>();
        cases.push((witness(&keys, 65, None), 0.0));
    }
    let (epsilon, smallest) = (T::epsilon().to_f64().unwrap(), T::min_positive_value().to_f64().unwrap());
    let mut mismatches = Vec::new();
    for (number, (case, slack)) in cases.iter().enumerate() {
        let label = format!("{:?} staged case {number} {}", T::data_type(), case.label());
        let (cpu, vulkan) = (case.cpu::<T>(), case.gpu::<T>(fixture, &case.vulkan_kernel::<T>(fixture)));
        for (index, (cpu, vulkan)) in cpu.iter().zip(&vulkan).enumerate() {
            let (c, v) = (cpu.to_f64().unwrap(), vulkan.to_f64().unwrap());
            let quotient = 3.0 * 2f64.powi(-23) * c.abs();
            // One storage step for BF16, whose rounding of FP32 values this close may differ by one.
            let storage = if size_of::<T>() == 2 {
                epsilon * (c.abs() + slack).max(smallest)
            } else {
                0.0
            };
            let same = match c.is_finite() {
                true => (c - v).abs() <= slack + quotient + storage,
                false => c == v || (c.is_nan() && v.is_nan()),
            };
            if !same {
                mismatches.push(format!("{label}: element {index}: CPU {c:e}, Vulkan {v:e}, slack {slack:e}"));
                break;
            }
        }
    }
    assert!(mismatches.is_empty(), "{} staged mismatches:\n{}", mismatches.len(), mismatches.join("\n"));
    // One visible key at score 0: the output is its value, bit for bit, subnormal or signed zero.
    let mut single = Case::new(64, 1, 1, 0, 1, 7);
    single.keys.iter_mut().filter(|key| !key.is_nan()).for_each(|key| *key = 0.0);
    let smallest = T::min_positive_value().to_f32().unwrap() * T::epsilon().to_f32().unwrap();
    for (j, value) in single.values.iter_mut().filter(|value| !value.is_nan()).enumerate() {
        *value = [smallest, -smallest * 3.0, T::min_positive_value().to_f32().unwrap() * 0.75, -0.0, 0.0][j % 5];
    }
    let (cpu, vulkan) = (single.cpu::<T>(), single.gpu::<T>(fixture, &single.vulkan_kernel::<T>(fixture)));
    let bits = |values: &[T]| bytemuck::cast_slice::<T, u8>(values).to_vec();
    let summed = single.values[..64].iter().map(|&value| value + 0.0).collect::<Vec<_>>();
    assert_eq!(bits(&cpu), bits(&Case::stored::<T>(&summed)), "{:?} single key CPU", T::data_type());
    assert_eq!(bits(&vulkan), bits(&cpu), "{:?} single key Vulkan", T::data_type());
}

#[uzu_test]
fn staged_subnormal_weights_and_values() {
    let fixture = KernelFixture::new();
    staged::<f32>(&fixture);
    staged::<bf16>(&fixture);
    fixture.assert_clean();
}

/// A decode over 4096 keys with scores across ±80, a 128-token prefill after 1024 cached tokens, and head
/// dimension 512 over 2000 keys.
#[uzu_test]
fn long_sequences() {
    let mut decode = Case::new(128, 4, 2, 4095, 1, 41);
    decode.scale = 20.0;
    let prefill = Case::new(64, 2, 1, 1024, 128, 42);
    let wide = Case::new(512, 2, 2, 1998, 2, 43);
    check_both("long", &[decode, prefill, wide]);
}

/// Construction rejects other head dimensions and data types; `encode` rejects a zero GQA factor, a suffix past the
/// sequence and a missing ring before recording anything.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    let new = |data_type, head_dim| {
        AttentionSinglePassVulkanKernel::new(&fixture.context, data_type, head_dim, false, false, true, false, false)
    };
    assert!(matches!(new(DataType::F32, 96), Err(Error::KernelPrecondition { .. })));
    assert!(matches!(new(DataType::F16, 64), Err(Error::KernelVariant { .. })));
    let kernel = new(DataType::F32, 64).expect("AttentionSinglePass");
    let untouched = fixture.buffer(&[0u32; 1024]);
    let mut encoding = fixture.encoding();
    for (gqa_factor, sequence_length, ring) in [(0, 4, None), (1, 2, None), (1, 4, ring(0, 1))] {
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the checks fail before recording.
            kernel.encode(
                (&untouched, 0..1024),
                (&untouched, 1024..2048),
                (&untouched, 2048..3072),
                (&untouched, 3072..4096),
                gqa_factor,
                sequence_length,
                64,
                64,
                64,
                64,
                ring,
                1.0,
                None,
                None,
                None,
                1,
                3,
                &mut encoding,
            )
        }));
        let message = result.expect_err("encode accepted");
        let message = message.downcast_ref::<String>().expect("message");
        assert!(message.contains("AttentionSinglePass"), "{message}");
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded nothing.
    assert!(unsafe { KernelFixture::read::<u32>(&untouched) }.iter().all(|&word| word == 0), "a rejected call wrote");
    fixture.assert_clean();
}

/// No query or no head records nothing: the guarded output stays untouched.
#[uzu_test]
fn zero_groups_record_nothing() {
    let fixture = KernelFixture::new();
    let mut no_heads = Case::new(64, 2, 1, 4, 2, 9);
    no_heads.num_heads = 0;
    for case in [Case::new(64, 2, 1, 4, 0, 8), no_heads] {
        assert!(case.gpu::<f32>(&fixture, &case.vulkan_kernel::<f32>(&fixture)).is_empty());
    }
    fixture.assert_clean();
}

/// Run alone: `cargo test ... attention_single_pass_test::throughput -- --ignored --nocapture`. Causal attention of 32
/// query heads over 8 KV heads in the model's cache layout: decodes over 256 to 16384 keys at every head dimension in
/// BF16, FP32 samples, and prefills of 16 to 512 queries after 1024 cached tokens. "Unique" counts the logical K and V
/// bytes once; "requested" counts the rows every query of every query head reads, from cache when it holds them.
/// Effective GB/s is unique bytes over GPU time, not a fraction of any bandwidth limit.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + NoUninit + AnyBitPattern>(
        fixture: &KernelFixture,
        head_dim: u32,
        prefix: u32,
        suffix: u32,
    ) {
        let case = Case::new(head_dim, 32, 4, prefix, suffix, 1).with_model_layout();
        let kernel = case.vulkan_kernel::<T>(fixture);
        let buffer = |values: &[f32]| fixture.buffer(&Case::stored::<T>(values));
        let (queries, keys, values) = (buffer(&case.queries), buffer(&case.keys), buffer(&case.values));
        let out = fixture.buffer(&vec![T::zero(); (suffix * 32 * head_dim) as usize]);
        let mut encodes = Vec::new();
        let (gpu, wall) = fixture.median_times(|encoding| {
            let start = Instant::now();
            let buffers = [&queries, &keys, &values, &out].map(|buffer| (buffer, 0..buffer.size()));
            // SAFETY: whole buffers of the case's rows, heads and queries; the output aliases nothing.
            unsafe { case.encode(&kernel, buffers, None, None, encoding) };
            encodes.push(start.elapsed());
        });
        encodes.sort();
        let encode: Duration = encodes[encodes.len() / 2];
        let row_bytes = 2 * u64::from(head_dim) * size_of::<T>() as u64;
        let bytes = 8 * u64::from(case.sequence_length()) * row_bytes;
        let requested = 32 * (0..suffix).map(|query| u64::from(prefix + query + 1)).sum::<u64>() * row_bytes;
        eprintln!(
            "MEASURE AttentionSinglePass {:?} D {head_dim} prefix {prefix} suffix {suffix}: unique K+V {bytes} B, \
             requested {requested} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s effective), encode \
             {encode:?}, wall {wall:?}",
            T::data_type(),
            bytes as f64 / gpu.as_secs_f64() / 1e9
        );
    }
    let fixture = KernelFixture::new();
    for (head_dim, keys) in itertools::iproduct!([64, 128, 256, 512], [256, 1024, 4096, 16384]) {
        measure::<bf16>(&fixture, head_dim, keys - 1, 1);
    }
    for keys in [1024, 16384] {
        measure::<f32>(&fixture, 128, keys - 1, 1);
    }
    for suffix in [16, 128, 512] {
        measure::<bf16>(&fixture, 128, 1024, suffix);
    }
    fixture.assert_clean();
}
