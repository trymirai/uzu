use std::collections::BTreeMap;

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{GemvCase as Case, kernel_fixture::KernelFixture};
use crate::{
    backends::vulkan::{Error, GemvVulkanKernel},
    data_type::DataType,
    tests::matmul::qwen3_layer_shapes,
};

const SCALE: u32 = 1;
const ACCUMULATE: u32 = 2;
const BIAS: u32 = 4;
const SOFT_CAP: u32 = 8;
const GATHER: u32 = 16;

/// Every B, A and D triple of F32 and BF16.
fn triples() -> impl Iterator<Item = [DataType; 3]> {
    itertools::iproduct!(
        [DataType::F32, DataType::BF16],
        [DataType::F32, DataType::BF16],
        [DataType::F32, DataType::BF16]
    )
    .map(|(b, a, d)| [b, a, d])
}

/// One case on the CPU and Vulkan against its FP64 bounds: where they exist both outputs within them, else Vulkan's
/// class and bits the CPU's (NaN any NaN), and under a soft cap the bounds of the soft cap at the CPU's exact FP32 input
/// where that is finite. Counts bounded and exact outputs and the largest Vulkan-CPU difference relative to the CPU's
/// magnitude (at least the smallest normal) per label.
fn check(
    fixture: &KernelFixture,
    label: &str,
    case: &Case,
    totals: &mut BTreeMap<String, [f64; 3]>,
) {
    let vulkan = case.gpu(fixture, &case.vulkan_kernel(fixture), 1);
    let cpu = case.cpu(1, false);
    let bounds = case.bounds();
    let before = (case.soft_cap.is_some() && bounds.iter().any(Option::is_none)).then(|| case.cpu(1, true));
    let total = totals.entry(label.to_owned()).or_default();
    for (index, bound) in bounds.into_iter().enumerate() {
        let name = format!("{label} {} output {index}", case.label());
        let bound = bound.or_else(|| {
            let value = f64::from(before.as_ref()?[index]);
            value.is_finite().then(|| Case::soft_cap_bounds((value, value), case.soft_cap?)).flatten()
        });
        match bound {
            Some(bound) => {
                for (backend, value) in [("CPU", cpu[index]), ("Vulkan", vulkan[index])] {
                    assert!(
                        case.within(value, bound),
                        "{name}: {backend} {value:e} outside [{:e}, {:e}]",
                        bound.0,
                        bound.1
                    );
                }
                total[0] += 1.0;
                let difference = f64::from(vulkan[index]) - f64::from(cpu[index]);
                total[2] =
                    total[2].max(difference.abs() / f64::from(cpu[index].abs()).max(f64::from(f32::MIN_POSITIVE)));
            },
            None => {
                KernelFixture::assert_bits(&cpu[index..=index], &vulkan[index..=index], &name);
                total[1] += 1.0;
            },
        }
    }
}

fn report(totals: &BTreeMap<String, [f64; 3]>) {
    for (label, [bounded, exact, difference]) in totals {
        eprintln!(
            "Gemv {label}: {bounded} bounded outputs, {exact} exact outputs, max relative |Vulkan - CPU| {difference:.3e}"
        );
    }
}

fn check_all(
    label: &str,
    cases: impl IntoIterator<Item = Case>,
) {
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    for case in cases {
        check(&fixture, label, &case, &mut totals);
    }
    report(&totals);
    fixture.assert_clean();
}

/// Both CPU and Vulkan store exactly `expected`, bit for bit (NaN any NaN).
fn exact(
    fixture: &KernelFixture,
    label: &str,
    case: &Case,
    expected: &[f32],
) {
    let vulkan = case.gpu(fixture, &case.vulkan_kernel(fixture), 1);
    KernelFixture::assert_bits(expected, &case.cpu(1, false), &format!("{label} CPU"));
    KernelFixture::assert_bits(expected, &vulkan, &format!("{label} Vulkan"));
}

/// One A row against `rows` of B under `mask`, without gather; other fields as `Case::new` makes them.
fn witness(
    types: [DataType; 3],
    a: &[f32],
    rows: &[&[f32]],
    mask: u32,
) -> Case {
    let mut case = Case::new(types, 1, rows.len() as u32, a.len() as u32, mask & !GATHER, 0);
    (case.a, case.b) = (a.to_vec(), rows.concat());
    case
}

/// Every triple under all 32 flag masks over 1 x 8 x 7 and 3 x 33 x 129 (m x n x k), gathered outputs repeating B rows
/// out of order, A starting 1 to 3 elements into its range.
#[uzu_test]
fn all_triples_and_masks() {
    let cases = itertools::iproduct!(triples(), 0..32, [(1, 8, 7), (3, 33, 129)])
        .map(|(types, mask, (m, n, k))| Case::new(types, m, n, k, mask, mask + m));
    check_all("masks", cases);
}

/// K tails 0, 1, 31 and 257 under no and every flag, 1 x 1 x 1, 2 x 1000 x 2048 and two Qwen3 layers at m 1 and 2.
#[uzu_test]
fn shapes_and_tails() {
    let mut cases = Vec::new();
    for (types, k, mask) in itertools::iproduct!(
        [[DataType::BF16; 3], [DataType::F32; 3], [DataType::BF16, DataType::F32, DataType::F32]],
        [0, 1, 31, 257],
        [0, 31]
    ) {
        cases.push(Case::new(types, 2, 9, k, mask, k));
    }
    cases.push(Case::new([DataType::BF16; 3], 1, 1, 1, SCALE | BIAS, 2));
    cases.push(Case::new([DataType::BF16, DataType::F32, DataType::F32], 2, 1000, 2048, ACCUMULATE | BIAS, 3));
    let layers = qwen3_layer_shapes(8).filter(|(label, shape)| ["0.8b_qkv", "4b_down"].contains(label) && shape.m <= 2);
    for ((_, shape), types) in
        itertools::iproduct!(layers, [[DataType::BF16; 3], [DataType::BF16, DataType::F32, DataType::F32]])
    {
        cases.push(Case::new(types, shape.m, shape.n, shape.k, SCALE | BIAS, shape.m));
    }
    check_all("shapes", cases);
}

/// No rows or no columns record nothing, leaving D's guards and inputs unchanged; K = 0 is the epilogue on +0.
#[uzu_test]
fn zero_dispatch_records_nothing() {
    let fixture = KernelFixture::new();
    for case in [Case::new([DataType::BF16; 3], 0, 5, 7, 31, 1), Case::new([DataType::F32; 3], 3, 0, 7, 31, 2)] {
        assert!(case.gpu(&fixture, &case.vulkan_kernel(&fixture), 1).is_empty(), "{}", case.label());
    }
    for types in triples() {
        let case = Case::new(types, 2, 3, 0, 0, 4);
        exact(&fixture, "K 0", &case, &[0.0; 6]);
    }
    fixture.assert_clean();
}

/// Two accumulating dispatches in one command buffer over exact integers, gathered: D + 2 A Bᵀ bit for bit.
#[uzu_test]
fn successive_accumulate() {
    let fixture = KernelFixture::new();
    let mut case = Case::new([DataType::BF16, DataType::BF16, DataType::F32], 3, 5, 40, ACCUMULATE | GATHER, 6);
    case.b = (0..case.b.len()).map(|index| (index % 7) as f32 - 3.0).collect();
    case.a = (0..case.a.len()).map(|index| (index % 5) as f32 - 2.0).collect();
    case.d = (0..case.d.len()).map(|index| index as f32).collect();
    let k = case.k as usize;
    let expected = (0..case.d.len())
        .map(|index| {
            let (row, weights) = (index / case.n as usize, &case.b[case.weight_row(index) * k..][..k]);
            let dot = case.a[row * k..][..k].iter().zip(weights).map(|(x, w)| x * w).sum::<f32>();
            case.d[index] + 2.0 * dot
        })
        .collect::<Vec<_>>();
    KernelFixture::assert_bits(&expected, &case.cpu(2, false), "CPU chain");
    KernelFixture::assert_bits(&expected, &case.gpu(&fixture, &case.vulkan_kernel(&fixture), 2), "Vulkan chain");
    fixture.assert_clean();
}

/// Exact results of normal, subnormal and signed-zero arithmetic, on the CPU and Vulkan bit for bit:
/// - integer dots 259, 257.5 and 257 storing BF16 260, 258 and 256 (ties to even) and F32 exactly;
/// - subnormal times huge: F32 2^-140 x 2^120 = 2^-20, BF16 2^-133 x 2^120 = 2^-13, which a flushed operand loses;
/// - 4096 products 2^-65 x 2^-65 summing to the normal 2^-118, each product subnormal;
/// - subnormal outputs: 2^-130 in F32 and BF16, 2^-140 in F32 and as BF16 +0;
/// - the epilogue: scale 2^-30 on 2^-100 is 2^-130, plus bias -2^-130 is +0; scale -1 on +0 is -0, plus D -0 stays
///   -0, plus bias +0 is +0;
/// - separately rounded steps: scale 1 + 2^-12 on a dot 1 + 2^-12 rounds to 1 + 2^-11, which D -(1 + 2^-11) cancels to
///   +0, where a fused multiply-add leaves 2^-24.
#[uzu_test]
fn exact_witnesses() {
    let fixture = KernelFixture::new();
    let two = |e: i32| 2f32.powi(e);
    let (f32s, bf16s) = ([DataType::F32; 3], [DataType::BF16; 3]);
    let integers: [&[f32]; 3] = [&[256.0, 2.0, 1.0], &[256.0, 1.0, 0.5], &[256.0, 1.0, 0.0]];
    exact(&fixture, "BF16 ties", &witness(bf16s, &[1.0; 3], &integers, 0), &[260.0, 258.0, 256.0]);
    exact(
        &fixture,
        "F32 integers",
        &witness([DataType::BF16, DataType::BF16, DataType::F32], &[1.0; 3], &integers, 0),
        &[259.0, 257.5, 257.0],
    );
    exact(&fixture, "F32 subnormal x huge", &witness(f32s, &[two(-140), 1.0], &[&[two(120), 0.0]], 0), &[two(-20)]);
    exact(&fixture, "BF16 subnormal x huge", &witness(bf16s, &[two(-133)], &[&[two(120)]], 0), &[two(-13)]);
    // Careful, ordinary, replayed and zero rows of A against one B row: a workgroup each in this geometry; only the
    // measured 4 x 8 microtile candidate puts all four in one workgroup.
    for (types, tiny, expected) in [(f32s, two(-140), two(-20)), (bf16s, two(-133), two(-13))] {
        let mut rows = Case::new(types, 4, 1, 1, 0, 0);
        (rows.a, rows.b) = (vec![tiny, 1.0, f32::INFINITY, 0.0], vec![two(120)]);
        exact(&fixture, "mixed rows", &rows, &[expected, two(120), f32::INFINITY, 0.0]);
    }
    for types in [f32s, bf16s] {
        exact(&fixture, "tiny products", &witness(types, &[two(-65); 4096], &[&[two(-65); 4096]], 0), &[two(-118)]);
    }
    exact(
        &fixture,
        "F32 subnormal outputs",
        &witness(f32s, &[two(-65)], &[&[two(-65)], &[two(-75)]], 0),
        &[two(-130), two(-140)],
    );
    exact(
        &fixture,
        "BF16 subnormal outputs",
        &witness(bf16s, &[two(-65)], &[&[two(-65)], &[two(-75)]], 0),
        &[two(-130), 0.0],
    );
    let mut scaled = witness(f32s, &[two(-50), 0.0], &[&[two(-50), 1.0], &[0.0, 3.0]], SCALE | BIAS);
    (scaled.ab_scale, scaled.bias) = (Some(two(-30)), Some(vec![0.0, 0.0]));
    exact(&fixture, "subnormal scale", &scaled, &[two(-130), 0.0]);
    scaled.bias = Some(vec![-two(-130), 0.0]);
    exact(&fixture, "subnormal cancellation", &scaled, &[0.0, 0.0]);
    let mut signs = witness(f32s, &[0.0], &[&[1.0]], SCALE);
    signs.ab_scale = Some(-1.0);
    exact(&fixture, "negated zero", &signs, &[-0.0]);
    let mut signs = witness(f32s, &[0.0], &[&[1.0]], SCALE | ACCUMULATE);
    (signs.ab_scale, signs.d) = (Some(-1.0), vec![-0.0]);
    exact(&fixture, "accumulated zero", &signs, &[-0.0]);
    signs = witness(f32s, &[0.0], &[&[1.0]], SCALE | ACCUMULATE | BIAS);
    (signs.ab_scale, signs.d, signs.bias) = (Some(-1.0), vec![-0.0], Some(vec![0.0]));
    exact(&fixture, "zero bias", &signs, &[0.0]);
    let mut rounding = witness(f32s, &[1.0 + two(-12)], &[&[1.0]], SCALE | ACCUMULATE);
    (rounding.ab_scale, rounding.d) = (Some(1.0 + two(-12)), vec![-(1.0 + two(-11))]);
    exact(&fixture, "no contraction", &rounding, &[0.0]);
    fixture.assert_clean();
}

/// The soft cap follows the bias: cap 1 on dot 0.5 plus bias 2 is tanh(2.5), within its bounds, where capping first
/// gives tanh(0.5) + 2.
#[uzu_test]
fn soft_cap_follows_bias() {
    let mut case = witness([DataType::F32; 3], &[0.5], &[&[1.0]], BIAS | SOFT_CAP);
    (case.bias, case.soft_cap) = (Some(vec![2.0]), Some(1.0));
    let bound = case.bounds()[0].expect("finite bounds");
    assert!(bound.0 > 0.98 && bound.1 < 0.99, "bounds {bound:?} are not tanh(2.5)");
    check_all("soft cap order", [case]);
}

/// The soft cap at zero, negative, tiny and nonfinite caps and inputs as IEEE gives them: infinite inputs give ±cap,
/// NaN NaN; cap 0 gives zeros of the input's sign, an infinite cap NaN; a negative cap within bounds; tiny quotients
/// keep every bit (tanh rounds to its argument): 2^-140 / 1, 2^-30 / 2^100 = 2^-130, and a subnormal cap 2^-140 with
/// 2^-130 / 2^-140 = 1024 saturating to exactly the cap.
#[uzu_test]
fn soft_cap_edges() {
    let fixture = KernelFixture::new();
    let two = |e: i32| 2f32.powi(e);
    let capped = |a: &[f32], b: &[f32], cap: f32| {
        let mut case = witness([DataType::F32; 3], a, &[b], SOFT_CAP);
        case.soft_cap = Some(cap);
        case
    };
    exact(&fixture, "infinite input", &capped(&[1.0, 1.0], &[f32::INFINITY, 1.0], 3.0), &[3.0]);
    exact(&fixture, "negative infinite input", &capped(&[-1.0], &[f32::INFINITY], 3.0), &[-3.0]);
    exact(&fixture, "NaN input", &capped(&[f32::NAN], &[1.0], 3.0), &[f32::NAN]);
    exact(&fixture, "zero cap", &capped(&[2.0], &[1.0], 0.0), &[0.0]);
    exact(&fixture, "zero cap, negative input", &capped(&[-2.0], &[1.0], 0.0), &[-0.0]);
    exact(&fixture, "zero cap, zero input", &capped(&[0.0], &[1.0], 0.0), &[f32::NAN]);
    exact(&fixture, "infinite cap", &capped(&[2.0], &[1.0], f32::INFINITY), &[f32::NAN]);
    exact(&fixture, "NaN cap", &capped(&[2.0], &[1.0], f32::NAN), &[f32::NAN]);
    exact(&fixture, "tiny quotient", &capped(&[two(-140)], &[1.0], 1.0), &[two(-140)]);
    exact(&fixture, "subnormal quotient", &capped(&[two(-30)], &[1.0], two(100)), &[two(-30)]);
    exact(&fixture, "subnormal cap", &capped(&[two(-130)], &[1.0], two(-140)), &[two(-140)]);
    let mut totals = BTreeMap::new();
    let caps =
        [(6.0, -3.0), (-0.3, -2.0), (2.0, two(-126)), (two(-120), two(-140)), (7.0, two(127)), (two(126), two(127))];
    for (value, cap) in caps {
        check(&fixture, "soft cap bounds", &capped(&[value], &[1.0], cap), &mut totals);
    }
    report(&totals);
    fixture.assert_clean();
}

/// Overflow and nonfinite operands, whose classes follow the CPU's order bit for bit:
/// - x = 1.5 2^127 summed x + x - x overflows to +inf, while x - x + x is x, the last term 32 products on (K = 33) so K
///   strides sum in another order;
/// - finite products 2^200 and -2^200 overflow to infinities of either sign and sum to NaN;
/// - the epilogue overflows: dot 2^127 times scale 2, FLT_MAX plus bias FLT_MAX, D +inf, bias NaN;
/// - infinities times subnormal, zero and normal operands, NaN operands of A and B;
/// - an intermediate overflow a soft cap would hide: products 2^24, 1, 0 ..., -2^24 (K = 33) cancel to 0 in the CPU's
///   order but to 1 in K strides, scale 2^127, D 2^127 and bias -2^127 then give 0, capped 0, where the strided sum
///   overflows to +inf and caps to 1, a finite value.
#[uzu_test]
fn overflow_and_nonfinite_witnesses() {
    let fixture = KernelFixture::new();
    let two = |e: i32| 2f32.powi(e);
    let f32s = [DataType::F32; 3];
    let x = 1.5 * two(127);
    let strided = |values: [f32; 3], k: usize| {
        let mut row = vec![0.0; k];
        (row[0], row[1], row[32]) = (values[0], values[1], values[2]);
        row
    };
    let (overflowing, finite) = (strided([x, x, -x], 33), strided([x, -x, x], 33));
    exact(&fixture, "overflow midpoint", &witness(f32s, &[1.0; 33], &[&overflowing, &finite], 0), &[f32::INFINITY, x]);
    exact(
        &fixture,
        "finite products to NaN",
        &witness(f32s, &[two(100), two(100)], &[&[two(100), -two(100)]], 0),
        &[f32::NAN],
    );
    let mut epilogue = witness(f32s, &[1.0], &[&[two(127)], &[f32::MAX]], SCALE | ACCUMULATE | BIAS);
    (epilogue.ab_scale, epilogue.d, epilogue.bias) = (Some(2.0), vec![0.0, 0.0], Some(vec![0.0, f32::MAX]));
    exact(&fixture, "epilogue overflow", &epilogue, &[f32::INFINITY, f32::INFINITY]);
    (epilogue.ab_scale, epilogue.d, epilogue.bias) =
        (Some(0.5), vec![f32::NEG_INFINITY, 1.0], Some(vec![1.0, f32::NAN]));
    exact(&fixture, "nonfinite epilogue", &epilogue, &[f32::NEG_INFINITY, f32::NAN]);
    let rows: [&[f32]; 4] = [&[two(-149), 1.0], &[0.0, 1.0], &[-2.0, 1.0], &[1.0, f32::NAN]];
    exact(
        &fixture,
        "infinite A",
        &witness(f32s, &[f32::INFINITY, 1.0], &rows, 0),
        &[f32::INFINITY, f32::NAN, f32::NEG_INFINITY, f32::NAN],
    );
    exact(&fixture, "NaN A", &witness(f32s, &[f32::NAN, two(-140)], &[&[0.0, 1.0]], 0), &[f32::NAN]);
    let bf16s = [DataType::BF16; 3];
    exact(
        &fixture,
        "BF16 infinite B",
        &witness(bf16s, &[two(-133), 0.0], &[&[f32::INFINITY, 1.0], &[1.0, f32::INFINITY]], 0),
        &[f32::INFINITY, f32::NAN],
    );
    let mut b = vec![0.0; 33];
    (b[0], b[1], b[32]) = (two(24), 1.0, -two(24));
    let mut hidden = witness(f32s, &[1.0; 33], &[&b], SCALE | ACCUMULATE | BIAS | SOFT_CAP);
    (hidden.ab_scale, hidden.d, hidden.bias, hidden.soft_cap) =
        (Some(two(127)), vec![two(127)], Some(vec![-two(127)]), Some(1.0));
    exact(&fixture, "soft cap hiding an overflow", &hidden, &[0.0]);
    // x + x - x - x, the negations 32 products on (K = 34), is exactly 0 but +inf in the CPU's order and 0 in K strides,
    // also scaled by 2^-20 or the subnormal 2^-140: the bounds must not claim a finite interval, leaving the CPU's class.
    let mut cancelling_row = strided([x, x, -x], 34);
    cancelling_row[33] = -x;
    let mut totals = BTreeMap::new();
    for scale in [None, Some(two(-20)), Some(two(-140))] {
        let mut cancelling = witness(
            f32s,
            &[1.0; 34],
            &[&cancelling_row],
            if scale.is_some() {
                SCALE
            } else {
                0
            },
        );
        cancelling.ab_scale = scale;
        assert!(cancelling.bounds()[0].is_none(), "finite bounds where the CPU's order overflows, scale {scale:?}");
        assert_eq!(cancelling.cpu(1, false), [f32::INFINITY], "CPU class, scale {scale:?}");
        check(&fixture, "cancelling overflow", &cancelling, &mut totals);
    }
    report(&totals);
    fixture.assert_clean();
}

/// The oracle's brackets, independent of the kernel: F32 values one step outside the bounds are rejected; BF16 bounds
/// round to nearest even from their exact FP64 value, normal or subnormal, of either sign: a bound just short of the
/// midpoint above an odd BF16 value keeps that value, where rounding through FP32 first reaches the midpoint and rounds on
/// to the even neighbour; a bound just past the midpoint above an even value rounds up, where half's FP64 conversion sees
/// a tie; and bounds from the midpoint above the largest finite value round to infinity.
#[uzu_test]
fn bracket_boundaries() {
    let f32s = witness([DataType::F32; 3], &[1.0], &[&[1.0]], 0);
    let bound = (1.0, 1.0 + 2f64.powi(-30));
    assert!(f32s.within(1.0, bound), "F32 value within the bounds rejected");
    for outside in [1f32.next_up(), 1f32.next_down()] {
        assert!(!f32s.within(outside, bound), "F32 neighbour {outside:e} outside the bounds accepted");
    }
    let bf16s = witness([DataType::BF16; 3], &[1.0], &[&[1.0]], 0);
    for (odd, step) in [(1.0078125, 2f64.powi(-7)), (2f64.powi(-133), 2f64.powi(-133))] {
        let short = (odd + step / 2.0) * (1.0 - 2f64.powi(-40));
        let even = (odd + step) as f32;
        assert!(bf16s.within(odd as f32, (odd, short)), "BF16 {odd:e} rejected");
        assert!(!bf16s.within(even, (odd, short)), "BF16 {even:e} past the midpoint accepted");
        assert!(bf16s.within(-odd as f32, (-short, -odd)), "BF16 {:e} rejected", -odd);
        assert!(!bf16s.within(-even, (-short, -odd)), "BF16 {:e} past the midpoint accepted", -even);
    }
    // Just above the midpoint over an even BF16 value, past it only in FP64's low 32 bits, rounds up to the odd one.
    for (even, step) in [(1.0, 2f64.powi(-7)), (2.0 * 2f64.powi(-133), 2f64.powi(-133))] {
        let above = (even + step / 2.0) * (1.0 + 2f64.powi(-40));
        let odd = (even + step) as f32;
        assert!(bf16s.within(odd, (above, odd.into())), "BF16 {odd:e} rejected");
        assert!(!bf16s.within(even as f32, (above, odd.into())), "BF16 {even:e} below the midpoint accepted");
        assert!(bf16s.within(-odd, (f64::from(-odd), -above)), "BF16 {:e} rejected", -odd);
        assert!(!bf16s.within(-even as f32, (f64::from(-odd), -above)), "BF16 {:e} below the midpoint accepted", -even);
    }
    // The largest finite BF16 value 2^128 - 2^120 is odd: below 2^128 - 2^119 values round to it, from there to infinity.
    let (largest, boundary) = (bf16::MAX.to_f32(), 2f64.powi(128) - 2f64.powi(119));
    for sign in [1.0f64, -1.0] {
        let [largest, infinity] = [largest, f32::INFINITY].map(|value| sign as f32 * value);
        let short = sign * (boundary - 2f64.powi(100));
        let span = |a: f64, b: f64| (a.min(b), a.max(b));
        assert!(bf16s.within(largest, span(f64::from(largest), short)), "{largest:e} rejected below the boundary");
        assert!(!bf16s.within(infinity, span(f64::from(largest), short)), "{infinity:e} accepted below the boundary");
        for beyond in [sign * boundary, sign * (boundary + 2f64.powi(100))] {
            assert!(bf16s.within(infinity, (beyond, beyond)), "{infinity:e} rejected at {beyond:e}");
            assert!(!bf16s.within(largest, (beyond, beyond)), "{largest:e} accepted at {beyond:e}");
        }
    }
}

/// Run alone: `cargo test ... gemv_test::throughput -- --ignored --nocapture`. Layers 0.8b_qkv, 0.8b_down, 2b_up, 4b_up
/// and 4b_down of qwen3_layer_shapes(8) at m 1, 2, 4 and 8, in BF16, F32, and BF16 weights with F32 input and output,
/// hashed data and no flags, each case first checked against the CPU and its bounds. Then the precision paths on 2b_up
/// at m 1 in BF16: ordinary data, A scaled by 2^-110 (every output's careful pass), zero A (no careful pass) and a NaN
/// in A (every output replayed in order), and its empty dispatch (m 0). GPU and wall are medians of 10 after 3 warm-up
/// submissions. Requested bytes count B once per row of A, unique bytes once if any row: logical rates, not DRAM
/// traffic; an empty dispatch requests nothing and has no rate.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    let whole = |buffer: &std::sync::Arc<crate::backends::vulkan::VkBuffer>| (buffer.clone(), 0..buffer.size());
    let measure = |label: &str, case: &Case| {
        let kernel = case.vulkan_kernel(&fixture);
        // Empty operands of an empty dispatch still need a buffer, which it never reads.
        let buffers = [(&case.b, 0), (&case.a, 1), (&case.d, 2)].map(|(values, index)| {
            let bytes = Case::bytes(values, case.types[index]);
            whole(&fixture.buffer(if bytes.is_empty() {
                &[0u8; 4][..]
            } else {
                &bytes
            }))
        });
        let (gpu, wall) = fixture.median_times(|encoding| {
            let [b, a, d] = buffers.each_ref().map(|(buffer, range)| (buffer, range.clone()));
            // SAFETY: whole buffers of the case's B, A and D; D aliases nothing.
            unsafe { case.encode(&kernel, [b, a, d], None, None, encoding) };
        });
        let [b, a, d] = case.types.map(|data_type| data_type.size_in_bytes() as u64);
        let (m, n, k) = (u64::from(case.m), u64::from(case.n), u64::from(case.k));
        let requested = m * n * k * b + m * k * a + m * n * d;
        let unique = u64::from(m > 0) * n * k * b + m * k * a + m * n * d;
        let rates = match m {
            0 => "no rate".to_owned(),
            _ => {
                let rate = |bytes: u64| bytes as f64 / gpu.as_secs_f64() / 1e9;
                format!("unique {:.1} GB/s, requested {:.1} GB/s", rate(unique), rate(requested))
            },
        };
        eprintln!(
            "MEASURE {label} m {m} {:?}: GPU {:.1} us, wall {:.1} us, unique {unique} B, requested {requested} B, {rates}",
            case.types,
            gpu.as_secs_f64() * 1e6,
            wall.as_secs_f64() * 1e6,
        );
    };
    let mut totals = BTreeMap::new();
    let layers = ["0.8b_qkv", "0.8b_down", "2b_up", "4b_up", "4b_down"];
    let types = [[DataType::BF16; 3], [DataType::F32; 3], [DataType::BF16, DataType::F32, DataType::F32]];
    for ((label, shape), types) in itertools::iproduct!(
        qwen3_layer_shapes(8).filter(|(label, shape)| layers.contains(label) && shape.m <= 8),
        types
    ) {
        let case = Case::new(types, shape.m, shape.n, shape.k, 0, shape.m);
        check(&fixture, "throughput", &case, &mut totals);
        measure(label, &case);
    }
    let shape = qwen3_layer_shapes(8).find(|(label, shape)| *label == "2b_up" && shape.m == 1).expect("2b_up").1;
    let ordinary = Case::new([DataType::BF16; 3], 1, shape.n, shape.k, 0, 1);
    let mut careful = ordinary.clone();
    careful.a.iter_mut().for_each(|value| *value = Case::stored(*value * 2f32.powi(-110), DataType::BF16));
    let mut zero = ordinary.clone();
    zero.a.fill(0.0);
    let mut replayed = ordinary.clone();
    replayed.a[0] = f32::NAN;
    for (path, case) in [("ordinary", &ordinary), ("careful", &careful), ("zero A", &zero), ("replay", &replayed)] {
        check(&fixture, &format!("path {path}"), case, &mut totals);
        measure(&format!("2b_up path {path}"), case);
    }
    // No row of A: the binding records nothing, which zero_dispatch_records_nothing checks.
    measure("2b_up empty dispatch", &Case::new([DataType::BF16; 3], 0, shape.n, shape.k, 0, 1));
    report(&totals);
    fixture.assert_clean();
}

/// Construction rejects every data type but F32 and BF16.
#[uzu_test]
fn rejects_invalid_types() {
    let fixture = KernelFixture::new();
    for types in [[DataType::F16, DataType::F32, DataType::F32], [DataType::F32, DataType::F32, DataType::F16]] {
        let [a, b, d] = types;
        let kernel = GemvVulkanKernel::new(&fixture.context, a, b, d, false, false, false, false, false);
        assert!(matches!(kernel, Err(Error::KernelVariant { .. })), "{types:?} accepted");
    }
    fixture.assert_clean();
}
