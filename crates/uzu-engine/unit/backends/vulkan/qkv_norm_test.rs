use std::{
    fmt::Debug,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Instant,
};

use bytemuck::NoUninit;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{
    NormalizationCase, QKVNormCase as Case, check_bounds, kernel_fixture::KernelFixture, normalization_stage_oracle,
    round32, to,
};
use crate::{
    array::ArrayElement,
    backends::vulkan::{Error, QKVNormVulkanKernel},
    data_type::DataType,
    encodable_block::normalization::PostLayerScalar,
};

/// Runs `$check::<InputT, ScaleT, OutputT>()` for all 8 out-of-place storage type combinations.
macro_rules! for_each_combination {
    ($check:ident) => { for_each_combination!(@input $check [f32 bf16]); };
    (@input $check:ident [$($input:tt)*]) => { $(for_each_combination!(@scale $check $input [f32 bf16]);)* };
    (@scale $check:ident $input:tt [$($scale:tt)*]) => { $(for_each_combination!(@output $check $input $scale [f32 bf16]);)* };
    (@output $check:ident $input:tt $scale:tt [$($output:tt)*]) => { $($check::<$input, $scale, $output>();)* };
}

/// Bounds of the whole output from the selected elements' bounds: every other element keeps its initial value exactly.
fn scatter<I: ArrayElement + Float, S: ArrayElement + Float, O: Float>(
    case: &Case<I, S>,
    selected: Vec<((f64, f64), f64)>,
) -> Vec<((f64, f64), f64)> {
    let initial = case.initial_output::<O>().into_iter().map(|value| value.to_f64().unwrap());
    let mut bounds = initial.map(|value| ((value, value), value)).collect::<Vec<_>>();
    for (index, bound) in case.selected().into_iter().zip(selected) {
        bounds[index] = bound;
    }
    bounds
}

/// The canonical Normalization stage oracle of the selected heads, as bounds: FP64 statistics within its FP32 budget,
/// rounded to the output type exactly where the contract converts.
fn ordinary_bounds<I: ArrayElement + Float, S: ArrayElement + Float, O: ArrayElement + Float>(
    case: &Case<I, S>
) -> Vec<((f64, f64), f64)> {
    let normalization = case.normalization_case();
    let selected = match normalization.input.is_empty() {
        true => Vec::new(),
        false => {
            let (expected, allowance) = normalization_stage_oracle::<I, S, O>(&normalization, &normalization.input);
            let centers = expected.into_iter().map(|value| value.to_f64().unwrap());
            centers
                .zip(allowance)
                .map(|(center, allowance)| ((center - allowance, center + allowance), center))
                .collect()
        },
    };
    scatter::<I, S, O>(case, selected)
}

/// The gap above |value|'s binade, 2^(floor(log2 |value|) - 23): an FP32 ULP for normal values and the scaled
/// domain's for subnormal ones.
fn binade_ulp(value: f64) -> f64 {
    2f64.powi(value.abs().log2().floor() as i32 - 23)
}

/// FP32 endpoints of an operation whose exact result `value` the device may miss by `ulps` binade ULPs; exact zeros
/// and non-finite values stay. Rounding each widened endpoint to nearest never moves it past the representable values
/// the device may return.
fn widened(
    value: f64,
    ulps: f64,
) -> (f64, f64) {
    match value == 0.0 || !value.is_finite() {
        true => (value, value),
        false => (round32(value - ulps * binade_ulp(value)), round32(value + ulps * binade_ulp(value))),
    }
}

/// Bounds of an FP32 sum of `terms`, each an interval of FP32 values, in any order: the CPU's sequential sum or the
/// device's per-invocation sums over a workgroup of `threads`, either a tree of n - 1 additions within
/// γ(n - 1) Σ|t| of the exact sum, γ(k) = k u / (1 - k u), u = 2^-24, and exact where the terms are exact multiples of
/// some 2^k with Σ|t| < 2^(k + 24), since every partial sum is then representable. Where any invocation's partial may
/// reach 2^-101 the device sums natively and may flush a subnormal partial or result: at most 2 threads 2^-126 more;
/// otherwise it sums scaled, which is IEEE. Non-finite terms sum exactly to their class.
pub fn sum_bounds(
    terms: &[(f64, f64)],
    threads: usize,
) -> (f64, f64) {
    let magnitude = |&(lo, hi): &(f64, f64)| lo.abs().max(hi.abs());
    let (lo, hi) = (terms.iter().map(|t| t.0).sum::<f64>(), terms.iter().map(|t| t.1).sum::<f64>());
    if !lo.is_finite() || !hi.is_finite() {
        return (lo, hi);
    }
    let absolute = terms.iter().map(magnitude).sum::<f64>();
    let grain = terms.iter().filter(|t| t.0 != 0.0).map(|t| {
        let bits = (t.0 as f32).to_bits() & 0x7fff_ffff;
        let significand = bits & 0x7f_ffff
            | if bits >> 23 == 0 {
                0
            } else {
                0x80_0000
            };
        (bits >> 23).max(1) as i32 - 150 + significand.trailing_zeros() as i32
    });
    let degenerate = terms.iter().all(|t| t.0.to_bits() == t.1.to_bits());
    if degenerate && grain.min().is_none_or(|grain| absolute < 2f64.powi(grain + 24)) {
        return (lo, hi);
    }
    let n = terms.len() as f64;
    let gamma = (n - 1.0).max(0.0) * 2f64.powi(-24) / (1.0 - (n - 1.0).max(0.0) * 2f64.powi(-24));
    let native = (0..threads)
        .any(|thread| terms.iter().skip(thread).step_by(threads).map(magnitude).sum::<f64>() >= 2f64.powi(-101));
    let error = gamma * absolute
        + if native {
            2.0 * threads as f64 * 2f64.powi(-126)
        } else {
            0.0
        };
    (round32(lo - error), round32(hi + error))
}

/// Bounds of staged_mean's quotient of an FP32 sum by the FP32 count: zeros exact, otherwise within 2.5 binade ULPs.
pub fn mean_bounds(
    (lo, hi): (f64, f64),
    count: f64,
) -> (f64, f64) {
    let quotient = |sum: f64, end: usize| match sum == 0.0 {
        true => sum,
        false => [widened(sum / count, 2.5).0, widened(sum / count, 2.5).1][end],
    };
    (quotient(lo, 0), quotient(hi, 1))
}

/// Bounds of the reciprocal square root of an FP32 shifted variance: within Vulkan's 2 ULPs of positive finite values,
/// exact by class otherwise (zeros give infinities of their sign, +inf gives +0, NaN and negative values NaN).
pub fn reciprocal_root_bounds(shifted: f64) -> (f64, f64) {
    match shifted {
        _ if shifted.is_nan() || shifted < 0.0 => (f64::NAN, f64::NAN),
        0.0 => (f64::INFINITY.copysign(shifted), f64::INFINITY.copysign(shifted)),
        f64::INFINITY => (0.0, 0.0),
        _ => widened(1.0 / shifted.sqrt(), 2.0),
    }
}

/// Endpoints of the CPU's FP32 staging of each row of a plain RMS or LayerNorm Normalization `case` (no biases,
/// transform or post-layer scalar), as computed by the CPU and by a device workgroup of `threads`, for what the
/// ordinary FP64 allowance cannot represent: rounding-order-sensitive products, statistics that overflow, vanish or reach
/// subnormals, and their classes. Elementwise differences, squares, the epsilon sum and products are exact FP32
/// roundings keeping subnormals on both; sums follow `sum_bounds`, means `mean_bounds`, and the reciprocal square root
/// is within Vulkan's 2 ULPs of positive finite values, exact by class otherwise (zeros give infinities of their sign,
/// +inf gives +0, NaN and negative values NaN). LayerNorm deviations (x - pivot) - mean_delta, from the first element as
/// pivot, follow the mean's bounds. Later stages are monotonic roundings of each endpoint and corner, which must share a
/// class.
pub fn staged_rms_bounds<I: Float, S: Float, O: Float>(
    case: &NormalizationCase<I, S>,
    threads: usize,
) -> Vec<((f64, f64), f64)> {
    let plain = case.biases.is_none() && case.hadamard_factors.is_none();
    assert!(plain && matches!(case.post_layer_scalar, PostLayerScalar::None), "plain RMS or LayerNorm only");
    let n = case.element_count as usize;
    let count = f64::from(n as f32);
    let (epsilon, offset) = (f64::from(case.epsilon), f64::from(case.scale_offset));
    let mut bounds = Vec::new();
    for row in case.input.chunks(n.max(1)) {
        let x = row.iter().map(|value| value.to_f32().unwrap()).collect::<Vec<_>>();
        let deltas = match case.subtract_mean {
            true => x.iter().map(|&value| f64::from(value - x[0])).collect::<Vec<_>>(),
            false => x.iter().map(|&value| f64::from(value)).collect(),
        };
        let mean_delta = match case.subtract_mean {
            true => mean_bounds(sum_bounds(&deltas.iter().map(|&d| (d, d)).collect::<Vec<_>>(), threads), count),
            false => (0.0, 0.0),
        };
        let deviations = deltas
            .iter()
            .map(|&delta| (round32(delta - mean_delta.1), round32(delta - mean_delta.0)))
            .collect::<Vec<_>>();
        let squares = deviations
            .iter()
            .map(|&(lo, hi)| match lo <= 0.0 && 0.0 <= hi && lo != hi {
                true => (0.0, round32(lo * lo).max(round32(hi * hi))),
                false => (round32(lo * lo).min(round32(hi * hi)), round32(lo * lo).max(round32(hi * hi))),
            })
            .collect::<Vec<_>>();
        let (sum_lo, sum_hi) = sum_bounds(&squares, threads);
        let variance = mean_bounds(
            (
                if sum_lo < 0.0 {
                    0.0
                } else {
                    sum_lo
                },
                sum_hi,
            ),
            count,
        );
        let shifted = [round32(variance.0 + epsilon), round32(variance.1 + epsilon)];
        let roots = [reciprocal_root_bounds(shifted[1]).0, reciprocal_root_bounds(shifted[0]).1];
        for (i, &(deviation_lo, deviation_hi)) in deviations.iter().enumerate() {
            let stage = |normalized: f64| match &case.scales {
                None => to::<O>(normalized),
                Some(scales) => {
                    let scale = round32(scales[i].to_f64().unwrap() + offset);
                    match case.full_layer {
                        true => to::<O>(round32(normalized * scale)),
                        false => to::<O>(round32(to::<O>(normalized) * to::<O>(scale))),
                    }
                },
            };
            let corners = [deviation_lo, deviation_hi]
                .into_iter()
                .flat_map(|deviation| roots.map(|root| stage(round32(deviation * root))))
                .collect::<Vec<_>>();
            let nan = corners.iter().filter(|value| value.is_nan()).count();
            assert!(nan == 0 || nan == corners.len(), "row {x:?}: the staging's class is ambiguous");
            let (lo, hi) =
                corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            let center = match (nan > 0, corners.iter().all(|value| value.to_bits() == corners[0].to_bits())) {
                (true, _) => f64::NAN,
                (false, true) => corners[0],
                (false, false) => 0.5 * lo + 0.5 * hi,
            };
            bounds.push(((lo, hi), center));
        }
    }
    bounds
}

/// `staged_rms_bounds` of the selected heads, scattered over the whole output.
fn staged_bounds<I: ArrayElement + Float, S: ArrayElement + Float, O: ArrayElement + Float>(
    case: &Case<I, S>
) -> Vec<((f64, f64), f64)> {
    scatter::<I, S, O>(case, staged_rms_bounds::<I, S, O>(&case.normalization_case(), 128))
}

/// Runs every case with one kernel on Vulkan in one command buffer and on the CPU, checking both against `bounds`.
fn check<I, S, O>(
    fixture: &KernelFixture,
    cases: &[Case<I, S>],
    bounds: fn(&Case<I, S>) -> Vec<((f64, f64), f64)>,
) where
    I: ArrayElement + Float,
    S: ArrayElement + Float,
    O: ArrayElement + Float + NoUninit + Debug,
{
    let kernel = cases[0].vulkan_kernel::<O>(fixture);
    let outputs = Case::gpu::<O>(fixture, &kernel, cases, fixture.encoding());
    for (case, gpu) in cases.iter().zip(outputs) {
        check_bounds(&bounds(case), &case.cpu::<O>(), &gpu, &case.label::<O>());
    }
}

/// Query, key and value selections of packed gated rows, of one and three rows, over odd, small, tile and large head
/// widths, without scales, with full_layer scales and with only-normalization scales and an offset; empty selections
/// record nothing.
fn matches_oracle<I, S, O>()
where
    I: ArrayElement + Float,
    S: ArrayElement + Float,
    O: ArrayElement + Float + NoUninit + Debug,
{
    let fixture = KernelFixture::new();
    let modes: [fn(Case<I, S>) -> Case<I, S>; 3] =
        [|case| case, |case| case.scales(true, 0.0), |case| case.scales(false, 1.0)];
    for mode in modes {
        let mut cases = Vec::new();
        for (index, head_dim) in [1, 7, 64, 130, 256].into_iter().enumerate() {
            for selection in [(0, 4), (4, 1), (5, 1)] {
                cases.push(mode(Case::new(1 + 2 * (index as u32 % 2), 6, head_dim, selection, index)));
            }
        }
        cases.extend(
            [(0, 64, (0, 4)), (2, 64, (0, 0)), (2, 0, (0, 4))]
                .map(|(batch, head_dim, selection)| mode(Case::new(batch, 6, head_dim, selection, 1))),
        );
        check::<I, S, O>(&fixture, &cases, ordinary_bounds::<I, S, O>);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn matches_oracle_all_combinations() {
    for_each_combination!(matches_oracle);
}

/// The model's in-place use with each scale type: one type throughout, the selection read back before overwriting.
fn in_place<T: ArrayElement + Float + NoUninit + Debug, S: ArrayElement + Float>() {
    let fixture = KernelFixture::new();
    let modes: [fn(Case<T, S>) -> Case<T, S>; 3] =
        [|case| case.in_place(), |case| case.in_place().scales(true, 0.0), |case| case.in_place().scales(false, 1.0)];
    for mode in modes {
        let cases = [(1, 128, (0, 4)), (3, 128, (4, 1)), (3, 64, (5, 1)), (2, 300, (0, 6))]
            .map(|(batch, head_dim, selection)| mode(Case::new(batch, 6, head_dim, selection, batch as usize)));
        check::<T, S, T>(&fixture, &cases, ordinary_bounds::<T, S, T>);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn in_place_all_types() {
    in_place::<f32, f32>();
    in_place::<f32, bf16>();
    in_place::<bf16, f32>();
    in_place::<bf16, bf16>();
}

/// The model's packed sequence on one in-place BF16 buffer in one command buffer: query, key and value selections with
/// full_layer alternating true, false, true on the same pipeline, each against its own staging, the gate untouched.
#[uzu_test]
fn full_layer_alternates_within_one_command_buffer() {
    let fixture = KernelFixture::new();
    let base = Case::<bf16, f32>::new(3, 6, 128, (0, 4), 9).in_place().scales(true, 1.0);
    let kernel = base.vulkan_kernel::<bf16>(&fixture);
    let selections = [((0, 4), true), ((4, 1), false), ((5, 1), true)];
    let output = fixture.guarded(&base.initial_output::<bf16>(), bf16::from_f32(-7.0));
    let scales = fixture.buffer(base.scales.as_ref().unwrap());
    let mut encoding = fixture.encoding();
    for ((head_offset, head_count), full_layer) in selections {
        // SAFETY: the output holds the case's rows; in place, one type.
        unsafe {
            kernel.encode(
                None,
                Some((&scales, 0..scales.size())),
                (&output.0, output.1.clone()),
                base.batch_size,
                base.input_row_stride,
                base.head_dim,
                base.epsilon,
                base.scale_offset,
                head_offset,
                head_count,
                full_layer,
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    // SAFETY: the only command buffer using this buffer has completed.
    let gpu = unsafe { KernelFixture::read_guarded(&output, bf16::from_f32(-7.0)) };
    let mut cpu = base.input.clone();
    let mut bounds = scatter::<bf16, f32, bf16>(&base, Vec::new());
    for ((head_offset, head_count), full_layer) in selections {
        let case = Case {
            input: cpu,
            head_offset,
            head_count,
            full_layer,
            ..base.clone()
        };
        let selected = case.selected();
        let staged = ordinary_bounds::<bf16, f32, bf16>(&case);
        selected.iter().for_each(|&index| bounds[index] = staged[index]);
        cpu = case.cpu::<bf16>();
    }
    check_bounds(&bounds, &cpu, &gpu, "QKVNorm alternating full_layer");
    fixture.assert_clean();
}

/// Rounding-order-sensitive cases: FP32 inputs normalized against BF16 outputs, where the staged bounds of full_layer
/// and of only-normalization scaling are disjoint for many elements, so either kernel computing the other order, or
/// rounding the FP32 product early, fails; each order is then checked within its own narrow staged bounds.
#[uzu_test]
fn rounding_order_is_canonical() {
    let fixture = KernelFixture::new();
    for full_layer in [true, false] {
        let case = Case::<f32, f32>::new(3, 6, 256, (0, 6), 4).scales(full_layer, 0.37);
        let flipped = Case {
            full_layer: !full_layer,
            ..case.clone()
        };
        let (own, other) = (staged_bounds::<f32, f32, bf16>(&case), staged_bounds::<f32, f32, bf16>(&flipped));
        let separated = |(((lo, hi), _), ((other_lo, other_hi), _)): (&((f64, f64), f64), &((f64, f64), f64))| {
            hi < other_lo || other_hi < lo
        };
        let disjoint = own.iter().zip(&other).filter(|pair| separated(*pair)).count();
        assert!(disjoint >= 64, "full_layer {full_layer}: only {disjoint} elements separate the rounding orders");
        check::<f32, f32, bf16>(&fixture, &[case], staged_bounds::<f32, f32, bf16>);
    }
    fixture.assert_clean();
}

/// Statistics classes at normal magnitudes with the canonical epsilon: a zero head (zero outputs of each sign), a head
/// whose squares overflow (an infinite sum, zero outputs), one NaN or infinite element (NaN or zero heads), subnormal
/// elements of an ordinary head kept through the exact products, and a large epsilon dominating the mean.
fn special_heads_match_staging<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let heads: [&[f32]; 6] = [
        &[0.0, -0.0, 0.0, -0.0],
        &[3e19, -2e19, 1.0, -0.0],
        &[1.0, f32::NAN, -2.0, 0.5],
        &[1.0, f32::INFINITY, -2.0, -0.0],
        &[1.5, f32::from_bits(0x0040_0000), -2.0, f32::from_bits(0x8000_0100)],
        &[1e-3, -2e-3, 3e-3, 0.0],
    ];
    for (full_layer, scale_offset) in [(true, 0.0), (false, 1.0)] {
        let mut case =
            Case::<T, T>::new(1, heads.len() as u32, 4, (0, heads.len() as u32), 1).scales(full_layer, scale_offset);
        for (head, values) in heads.iter().enumerate() {
            for (i, &value) in values.iter().enumerate() {
                case.input[head * 4 + i] = T::from(value).unwrap();
            }
        }
        let mut large_epsilon = case.clone();
        large_epsilon.epsilon = 4.0;
        check::<T, T, T>(&fixture, &[case, large_epsilon], staged_bounds::<T, T, T>);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn special_heads_match_staging_all_types() {
    special_heads_match_staging::<f32>();
    special_heads_match_staging::<bf16>();
}

/// One head per entry of `heads`, all `heads[0].len()` wide, with `epsilon`, full_layer scales and FP32 throughout.
fn statistics_case(
    heads: &[&[f32]],
    epsilon: f32,
) -> Case<f32, f32> {
    let head_dim = heads[0].len();
    let mut case = Case::<f32, f32>::new(1, heads.len() as u32, head_dim as u32, (0, heads.len() as u32), 1);
    case = case.scales(true, 0.0);
    case.epsilon = epsilon;
    for (head, values) in heads.iter().enumerate() {
        case.input[head * head_dim..(head + 1) * head_dim].copy_from_slice(values);
    }
    case
}

/// The CPU's FP32 mean of the squares of a head, for asserting which stage a case reaches.
fn cpu_mean(head: &[f32]) -> f32 {
    head.iter().fold(0f32, |sum, x| sum + x * x) / head.len() as f32
}

/// Statistics the CPU keeps subnormal or classifies, which flushing would turn into infinities, NaN or shifted normal
/// outputs: heads whose squares, sum or mean are subnormal under a zero, subnormal, negative-zero, negative, NaN or
/// infinite epsilon; a normal head under a subnormal epsilon; a head mixing normal and flushed-size squares, summed
/// natively; a mean plus epsilon cancelling to a tiny positive value; and sparse sums of subnormal units divided by odd
/// and even counts 3, 5, 7 and 130 near half units, a normal sum whose mean is subnormal and one whose mean is the
/// smallest normal. CPU and Vulkan are each checked against the staged endpoints.
#[uzu_test]
fn vanishing_statistics_match_staging() {
    let fixture = KernelFixture::new();
    let two = |exponent: i32| 2f32.powi(exponent);
    let (unit, two_units, three_units) = (3.0 * two(-76), two(-74), 5.0 * two(-76));
    assert_eq!([unit * unit, two_units * two_units, three_units * three_units].map(f32::to_bits), [1, 2, 3]);
    let tiny: &[f32] = &[two(-70), -1.5 * two(-70), 0.75 * two(-69), two(-71)];
    let normal_squares: &[f32] = &[two(-62), -two(-62), two(-62), -1.25 * two(-62)];
    let zero: &[f32] = &[0.0, -0.0, 0.0, 0.0];
    let tinier: &[f32] = &[1.5 * two(-75), -two(-74), 0.0, two(-76)];
    let mixed: &[f32] = &[1.0, two(-70), -1.5 * two(-75), unit];
    let mut cases = [0.0, f32::from_bits(0x0001_0000), -0.0, -1e-3, f32::NAN, f32::INFINITY]
        .map(|epsilon| statistics_case(&[tiny, normal_squares, zero, tinier, mixed], epsilon))
        .to_vec();
    cases.push(statistics_case(&[&[1.0; 4], &[-1.0, 1.0, -1.0, 1.0]], -(1.0 - two(-10))));
    let sparse = |count: usize, squares: &[f32]| {
        let mut head = vec![0.0; count];
        head[..squares.len()].copy_from_slice(squares);
        head
    };
    let sparse_heads = [
        (3, sparse(3, &[three_units, two_units])),
        (5, sparse(5, &[three_units, three_units, two_units])),
        (7, sparse(7, &[unit; 4])),
        (130, sparse(130, &[two_units; 98])),
        (130, sparse(130, &[two_units; 97])),
        (130, sparse(130, &[two(-63); 130])),
        (130, sparse(130, &[two(-63); 129])),
    ];
    for (count, head) in &sparse_heads {
        assert_eq!(head.len(), *count);
        let mean = cpu_mean(head);
        assert!(mean <= f32::MIN_POSITIVE, "count {count}: mean {mean:e} is not at or below the smallest normal");
        cases.push(statistics_case(&[head], 0.0));
    }
    // Two elements per invocation (256 of 128), around accumulate_squares' guards: partials of 1.125 and 0.5 times
    // 2^-101 followed by a square the device may flush, and squares of |x| = 2^-50 and its lower neighbor after a
    // normal or subnormal partial.
    let mut boundary = vec![0.0f32; 256];
    let below = |x: f32| f32::from_bits(x.to_bits() - 1);
    let pairs = [
        (1.5 * two(-51), two(-70)),
        (two(-51), below(two(-50))),
        (two(-51), two(-50)),
        (two(-70), two(-50)),
        (two(-70), below(two(-50))),
        (1.5 * two(-51), -two(-75)),
    ];
    for (thread, (first, second)) in pairs.into_iter().enumerate() {
        (boundary[thread], boundary[thread + 128]) = (first, second);
    }
    assert!((1.5 * two(-51)).powi(2) >= two(-101) && two(-51).powi(2) < two(-101) && two(-70).powi(2) < two(-126));
    cases.push(statistics_case(&[&boundary], 0.0));
    // Three elements per invocation (384 of 128): partials exactly 2^-101 and its predecessor before a third square
    // the device may flush, taking the native and the exact path.
    let mut threshold = vec![0.0f32; 384];
    (threshold[0], threshold[128], threshold[256]) = (two(-51), two(-51), two(-70));
    (threshold[1], threshold[129], threshold[257]) = (below(two(-51)), two(-51), -two(-70));
    let partial = |a: f32, b: f32| (a * a + b * b).to_bits();
    assert_eq!([partial(two(-51), two(-51)), partial(below(two(-51)), two(-51))], [0x0d00_0000, 0x0cff_ffff]);
    cases.push(statistics_case(&[&threshold], 0.0));
    assert_eq!(cpu_mean(&sparse_heads[5].1), f32::MIN_POSITIVE, "a mean of exactly the smallest normal");
    assert!(
        cpu_mean(&sparse_heads[6].1).is_subnormal() && sparse_heads[6].1.iter().map(|x| x * x).sum::<f32>().is_normal()
    );
    // Every output is finite but the zero head's under the zero epsilon (zero times an infinite reciprocal RMS).
    for (case, expected_finite) in cases.iter().zip([16, 20]) {
        let expected = staged_bounds::<f32, f32, f32>(case);
        let finite = case.selected().into_iter().filter(|&index| expected[index].1.is_finite()).count();
        assert_eq!(finite, expected_finite, "{}: the staging keeps {finite} finite outputs", case.label::<f32>());
    }
    for case in cases {
        check::<f32, f32, f32>(&fixture, &[case], staged_bounds::<f32, f32, f32>);
    }
    fixture.assert_clean();
}

/// Construction rejects F16 and I32, and in place with different input and output types before creating anything.
/// `encode` rejects a missing or extra input or scales, also for empty selections, and a row stride below the end of
/// active heads (an offset past the row, an offset plus count past it), without overflowing; empty selections with any
/// stride and offset pass and record nothing. The same command buffer then completes valid work ending exactly at the
/// row's last whole head, and rejected outputs stay untouched.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    let new = |types: [DataType; 3], in_place| {
        QKVNormVulkanKernel::new(&fixture.context, types[0], types[1], types[2], DataType::F32, in_place, true)
    };
    for types in [[DataType::F16, DataType::F32, DataType::F32], [DataType::F32, DataType::I32, DataType::F32]] {
        assert!(
            matches!(
                new(types, false),
                Err(Error::KernelVariant {
                    kernel: "QKVNorm",
                    ..
                })
            ),
            "{types:?}"
        );
    }
    for (input, output) in [(DataType::F32, DataType::BF16), (DataType::BF16, DataType::F32)] {
        for scale in [DataType::F32, DataType::BF16] {
            assert!(
                matches!(
                    new([input, scale, output], true),
                    Err(Error::KernelPrecondition {
                        kernel: "QKVNorm",
                        ..
                    })
                ),
                "in place {input:?} -> {output:?}"
            );
            assert!(new([input, scale, output], false).is_ok(), "out of place {input:?} -> {output:?}");
        }
    }
    let untouched = fixture.buffer(&[0u32; 1024]);
    let buffer = |index: u64| (&untouched, index * 1024..(index + 1) * 1024);
    let kernels = [false, true].map(|in_place| new([DataType::F32; 3], in_place).expect("QKVNorm"));
    let mut encoding = fixture.encoding();
    // (in_place kernel, input present, scales present, batch, stride, head_dim, head_offset, head_count, message)
    let span = "head_offset <= input_row_stride / head_dim && head_count <= input_row_stride / head_dim - head_offset";
    let rejected = [
        (0, false, true, 2, 256, 64, 0, 4, "must be present exactly when"),
        (0, true, false, 0, 256, 64, 0, 4, "must be present exactly when"),
        (1, true, true, 2, 256, 64, 0, 4, "must be present exactly when"),
        (1, false, false, 0, 0, 0, 0, 0, "must be present exactly when"),
        (0, true, true, 2, 255, 64, 0, 4, span),
        (1, false, true, 1, 0, 64, 0, 1, span),
        (0, true, true, 2, 256, 64, 5, 1, span),
        (0, true, true, 2, 256, 64, 2, 3, span),
        (0, true, true, 3, u32::MAX, u32::MAX, 0, u32::MAX, span),
        (0, true, true, 3, u32::MAX, 1, u32::MAX, u32::MAX, span),
        (1, false, true, 1, u32::MAX, 1, u32::MAX - 1, 2, span),
    ];
    for (kernel, input, scales, batch, stride, head_dim, head_offset, head_count, message) in rejected {
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: an assertion fails before recording.
            kernels[kernel].encode(
                input.then(|| buffer(0)),
                scales.then(|| buffer(1)),
                buffer(2),
                batch,
                stride,
                head_dim,
                1e-5,
                0.0,
                head_offset,
                head_count,
                true,
                &mut encoding,
            );
        }));
        let payload = result.expect_err("encode accepted");
        let text = payload.downcast_ref::<String>().expect("assertion message");
        let shape = format!("batch {batch} stride {stride} head_dim {head_dim} heads {head_offset}+{head_count}");
        assert!(text.contains(message), "{shape}: {text}");
    }
    for (batch, stride, head_dim, head_offset, head_count) in
        [(0, 0, 64, 9, 4), (2, 0, 64, u32::MAX, 0), (2, 0, 0, 9, 4), (1, 0, 0, u32::MAX, 0)]
    {
        // SAFETY: an empty selection records nothing.
        unsafe {
            kernels[1].encode(
                None,
                Some(buffer(1)),
                buffer(2),
                batch,
                stride,
                head_dim,
                1e-5,
                0.0,
                head_offset,
                head_count,
                false,
                &mut encoding,
            )
        };
    }
    // Rows of 8 whole heads (6 heads, a two-head gate and 3 padding elements): heads 5 to 7 end at the bound.
    let valid = Case::<f32, f32>::new(2, 6, 64, (5, 3), 2).scales(true, 0.0);
    assert_eq!(valid.input_row_stride / valid.head_dim, 8, "the valid selection ends at the row's last whole head");
    let gpu = Case::gpu::<f32>(&fixture, &kernels[0], std::slice::from_ref(&valid), encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    assert!(unsafe { KernelFixture::read::<u32>(&untouched) }.iter().all(|&word| word == 0), "a rejected call wrote");
    check_bounds(&ordinary_bounds::<f32, f32, f32>(&valid), &valid.cpu::<f32>(), &gpu[0], "valid work");
    fixture.assert_clean();
}

/// Run alone: `cargo test ... qkv_norm_test::throughput -- --ignored --nocapture`. Construction cost, then the model's
/// in-place query, key and value dispatches over decode and prefill batches of 32 query and 8 KV heads of 128.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + NoUninit>(fixture: &KernelFixture) {
        let mut construction = (0..11)
            .map(|_| {
                let start = Instant::now();
                Case::<T, f32>::new(1, 1, 1, (0, 1), 1).in_place().scales(true, 0.0).vulkan_kernel::<T>(fixture);
                start.elapsed()
            })
            .collect::<Vec<_>>();
        let first = construction[0];
        construction.sort();
        eprintln!("QKVNorm {:?} construction: first {first:?}, median of 11 {:?}", T::data_type(), construction[5]);
        for batch in [1u32, 128, 1024] {
            let case = Case::<T, f32>::new(batch, 48, 128, (0, 32), 1).in_place().scales(true, 0.0);
            let kernel = case.vulkan_kernel::<T>(fixture);
            let (rows, scales) = (fixture.buffer(&case.input), fixture.buffer(case.scales.as_ref().unwrap()));
            let (gpu, wall) = fixture.median_times(|encoding| {
                for (head_offset, head_count) in [(0, 32), (32, 8), (40, 8)] {
                    // SAFETY: the rows hold the case's payload and the scales one head; in place, one type.
                    unsafe {
                        kernel.encode(
                            None,
                            Some((&scales, 0..scales.size())),
                            (&rows, 0..rows.size()),
                            batch,
                            case.input_row_stride,
                            128,
                            case.epsilon,
                            0.0,
                            head_offset,
                            head_count,
                            true,
                            encoding,
                        );
                    }
                }
            });
            let bytes = 2 * u64::from(batch * 48 * 128) * size_of::<T>() as u64;
            eprintln!(
                "MEASURE QKVNorm {:?} q/k/v batch {batch}: {bytes} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?}",
                T::data_type(),
                bytes as f64 / gpu.as_secs_f64() / 1e9
            );
        }
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
