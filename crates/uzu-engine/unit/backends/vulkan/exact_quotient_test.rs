use std::{collections::BTreeSet, ops::Range, sync::Arc, time::Instant};

use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu_engine_macros::uzu_test;

use super::{arg, kernel_fixture::KernelFixture};
use crate::backends::vulkan::{VkBuffer, vk_kernels::TestExactQuotientVulkanKernel};

const SENTINEL: u32 = 0xa5a5_a5a5;

/// Pairs per GPU submission.
const CHUNK: usize = 1 << 16;

const INFINITY: u32 = 0x7f80_0000;

const QUIET_NAN: u32 = 0x7fc0_0000;

/// `(case, [a, b], a / b rounded to nearest even)`, frozen by hand: subnormal quotients on and off a midpoint, the
/// smallest normal over 1.5 onto the subnormal grid, and overflow by rounding, by doubling and by a subnormal divisor.
const WITNESSES: [(&str, [u32; 2], u32); 7] = [
    ("1.5 2^-149 ties to 2^-148", [0x0000_0003, 0x4000_0000], 0x0000_0002),
    ("2^-150 ties to +0", [0x0000_0001, 0x4000_0000], 0x0000_0000),
    ("0.75 2^-149 rounds to 2^-149", [0x0000_0003, 0x4080_0000], 0x0000_0001),
    ("2^-126 / 1.5 rounds down onto the subnormal grid", [0x0080_0000, 0x3fc0_0000], 0x0055_5555),
    ("max finite / (1 - 2^-24) is 2^128", [0x7f7f_ffff, 0x3f7f_ffff], INFINITY),
    ("max finite / 0.5 overflows", [0x7f7f_ffff, 0x3f00_0000], INFINITY),
    ("1 / 2^-140 overflows", [0x3f80_0000, 0x0000_0200], INFINITY),
];

/// a / b rounded to nearest even by exact rational arithmetic, independently of native division. Finite nonzero
/// operands are integers M 2^e with M below 2^24 and e = max(field, 1) - 150, so |a / b| = (Ma / Mb) 2^(ea - eb). Its
/// top bit t, 2^t <= |a / b| < 2^(t + 1), is that of Q = floor(Ma 2^64 / Mb) >= 2^40 moved by ea - eb - 64. From
/// t = 128 on it overflows; below t = -150 it is under 2^-150, half the smallest subnormal, and rounds to zero.
/// Otherwise the grid 2^g, g = max(t - 23, -149), puts the exact Ma 2^(ea - eb - g) / Mb in [2^-1, 2^24): a scaled
/// numerator stays below 2^48 and a scaled denominator at most 2 Ma < 2^25, so no u128 shift reaches 48 places. The
/// integer quotient n rounds up when twice the remainder exceeds the denominator, or equals it at an odd n; n 2^g
/// encodes as ((g + 149) << 23) + n, a carry stepping into the next exponent, at most into infinity.
fn rational_quotient(
    a: u32,
    b: u32,
) -> u32 {
    let sign = (a ^ b) & 0x8000_0000;
    let (a_magnitude, b_magnitude) = (a & 0x7fff_ffff, b & 0x7fff_ffff);
    match (a_magnitude, b_magnitude) {
        _ if a_magnitude > INFINITY || b_magnitude > INFINITY => return QUIET_NAN,
        (0, 0) | (INFINITY, INFINITY) => return QUIET_NAN,
        (INFINITY, _) | (_, 0) => return sign | INFINITY,
        (0, _) | (_, INFINITY) => return sign,
        _ => {},
    }
    let decode = |magnitude: u32| match magnitude >> 23 {
        0 => (u128::from(magnitude), -149),
        field => (u128::from(magnitude & 0x7f_ffff | 0x80_0000), field as i32 - 150),
    };
    let ((a_units, a_exponent), (b_units, b_exponent)) = (decode(a_magnitude), decode(b_magnitude));
    let scale = a_exponent - b_exponent;
    let top = (127 - ((a_units << 64) / b_units).leading_zeros()) as i32 - 64 + scale;
    if top >= 128 {
        return sign | INFINITY;
    }
    if top < -150 {
        return sign;
    }
    let grid = (top - 23).max(-149);
    let shift = scale - grid;
    let (numerator, denominator) = if shift >= 0 {
        (a_units << shift, b_units)
    } else {
        (a_units, b_units << -shift)
    };
    let (quotient, remainder) = (numerator / denominator, numerator % denominator);
    let up = 2 * remainder > denominator || (2 * remainder == denominator && quotient % 2 == 1);
    sign | ((((grid + 149) as u32) << 23) + (quotient + u128::from(up)) as u32)
}

/// Asserts `actual` is `expected` bit for bit, any NaN matching any NaN, naming both operands' bits.
fn assert_quotient(
    [a, b]: [u32; 2],
    expected: f32,
    actual: f32,
    case: &str,
) {
    let same = expected.to_bits() == actual.to_bits() || (expected.is_nan() && actual.is_nan());
    let (expected, actual) = (expected.to_bits(), actual.to_bits());
    assert!(same, "{case} [{a:#010x}, {b:#010x}]: expected {expected:#010x}, actual {actual:#010x}");
}

/// Asserts native division (expected) and the rational oracle (actual) agree on `[a, b]` bits and returns the quotient.
fn agree(pair: [u32; 2]) -> f32 {
    let native = f32::from_bits(pair[0]) / f32::from_bits(pair[1]);
    assert_quotient(pair, native, f32::from_bits(rational_quotient(pair[0], pair[1])), "native vs rational");
    native
}

/// The bits of ±significand 2^(exponent - 23) for a significand in [2^23, 2^24) and an exponent from -149 to 127; below
/// -126 the significand is truncated onto the subnormal grid, keeping its leading bit.
fn finite(
    sign: u32,
    exponent: i32,
    significand: u32,
) -> u32 {
    sign | if exponent < -126 {
        significand >> (-126 - exponent)
    } else {
        ((exponent + 127) as u32) << 23 | significand & 0x7f_ffff
    }
}

/// The exponent e of finite nonzero bits, 2^e <= |x| < 2^(e + 1).
fn exponent(bits: u32) -> i32 {
    match (bits >> 23) & 0xff {
        0 => (31 - (bits & 0x7f_ffff).leading_zeros()) as i32 - 149,
        field => field as i32 - 127,
    }
}

/// The 294 divisors: ±0, ±infinity, ±quiet NaN; 1 + 2^-23, 1 - 2^-24, 2 - 2^-23; 3, 7, 0.7, 1.1, ±1.3; max finite and
/// the largest subnormal; every power of two from 2^-149 to 2^127.
fn divisors() -> Vec<u32> {
    let mut divisors = vec![0x0000_0000, 0x8000_0000, INFINITY, 0xff80_0000, QUIET_NAN, 0xffc0_0000];
    divisors.extend([0x3f80_0001, 0x3f7f_ffff, 0x3fff_ffff]);
    divisors.extend([3.0f32, 7.0, 0.7, 1.1, 1.3, -1.3].map(f32::to_bits));
    divisors.extend([0x7f7f_ffff, 0x007f_ffff]);
    divisors.extend((-149..=127).map(|exponent| finite(0, exponent, 1 << 23)));
    assert_eq!(divisors.len(), 294, "divisors");
    assert_eq!(divisors.iter().collect::<BTreeSet<_>>().len(), 294, "distinct divisors");
    divisors
}

/// Every BF16 pattern widened by `half`, which must keep its bits exactly (a NaN its class), over every divisor.
fn bf16_by_divisors(visit: &mut dyn FnMut([u32; 2])) {
    let divisors = divisors();
    for bits in 0..=u16::MAX {
        let (widened, exact) = (bf16::from_bits(bits).to_f32(), f32::from_bits(u32::from(bits) << 16));
        assert!(widened.to_bits() == exact.to_bits() || (widened.is_nan() && exact.is_nan()), "BF16 {bits:#06x}");
        divisors.iter().for_each(|&divisor| visit([widened.to_bits(), divisor]));
    }
}

/// Each operand exponent difference (a's minus b's) from -151 to -126 and from 126 to 129 with 32768 seeded significand
/// pairs and signs. Even indices put the larger operand at exponent 127, odd ones the smaller at a subnormal exponent
/// from -149 to -127 with the other normal. The quotient's exponent is one less where a's significand is below b's.
fn exponent_strata(visit: &mut dyn FnMut([u32; 2])) {
    let mut rng = SmallRng::seed_from_u64(0x0d1f_5e1d);
    for difference in (-151..=-126i32).chain(126..=129) {
        for index in 0..1 << 15 {
            let smaller = if index % 2 == 0 {
                127 - difference.abs()
            } else {
                -127 - (index / 2) % 23
            };
            let exponents = if difference < 0 {
                [smaller, smaller - difference]
            } else {
                [smaller + difference, smaller]
            };
            let [a, b] = exponents
                .map(|exponent| finite(rng.random_range(0..2u32) << 31, exponent, rng.random_range(1 << 23..1 << 24)));
            assert_eq!(exponent(a) - exponent(b), difference, "[{a:#010x}, {b:#010x}] exponent difference");
            visit([a, b]);
        }
    }
}

/// Odd n 2^-149 for n from 1 to 65535 over 2^j for j from 1 to 24, onto the subnormal grid: a tie for j = 1 and where
/// n's low j bits are 2^(j - 1), otherwise off a midpoint. Bit 1 of n signs a and the parity of j signs b.
fn odd_subnormals(visit: &mut dyn FnMut([u32; 2])) {
    for n in (1..1u32 << 16).step_by(2) {
        for j in 1..=24u32 {
            visit([(n & 2) << 30 | n, (j & 1) << 31 | finite(0, j as i32, 1 << 23)]);
        }
    }
}

/// 2^20 seeded uniform bit pairs.
fn random_bits(visit: &mut dyn FnMut([u32; 2])) {
    let mut rng = SmallRng::seed_from_u64(0x0d1f_ba5e);
    for _ in 0..1 << 20 {
        visit([0; 2].map(|_| rng.random_range(..=u32::MAX)));
    }
}

/// Every family in order, each asserting its own count.
fn visit_families(mut visit: impl FnMut([u32; 2])) {
    let families: [(&str, usize, fn(&mut dyn FnMut([u32; 2]))); 5] = [
        ("BF16 numerators by divisors", 65536 * 294, bf16_by_divisors),
        ("operand exponent difference strata", 30 * 32768, exponent_strata),
        ("odd 2^-149 multiples over 2^j, subnormal grid", 32768 * 24, odd_subnormals),
        ("uniform bits", 1 << 20, random_bits),
        ("frozen witnesses", 7, |visit| WITNESSES.iter().for_each(|&(_, pair, _)| visit(pair))),
    ];
    for (family, expected, generate) in families {
        let mut count = 0;
        generate(&mut |pair| {
            count += 1;
            visit(pair);
        });
        assert_eq!(count, expected, "{family} count");
    }
}

/// Asserts a completed TestExactQuotient dispatch kept its guarded pairs and wrote `expected` between untouched guards.
///
/// # Safety
/// Every command buffer using the buffers has completed.
unsafe fn verify(
    (input, output): &((Arc<VkBuffer>, Range<u64>), (Arc<VkBuffer>, Range<u64>)),
    pairs: &[[u32; 2]],
    expected: &[f32],
) {
    // SAFETY: the caller guarantees completion.
    let actual = unsafe {
        KernelFixture::assert_unchanged(input, SENTINEL, pairs.as_flattened(), "pairs");
        KernelFixture::read_guarded::<u32>(output, SENTINEL)
    };
    assert_eq!(actual.len(), expected.len(), "TestExactQuotient length");
    for ((&pair, &expected), actual) in pairs.iter().zip(expected).zip(actual) {
        assert_quotient(pair, expected, f32::from_bits(actual), "TestExactQuotient");
    }
}

#[uzu_test]
fn frozen_witnesses_match_both_oracles() {
    for (case, pair, expected) in WITNESSES {
        assert_quotient(pair, f32::from_bits(expected), agree(pair), case);
    }
}

#[uzu_test]
fn host_families_agree() {
    visit_families(|pair| {
        agree(pair);
    });
}

/// Every family through TestExactQuotient in submissions of at most CHUNK pairs, each first checked by both host
/// oracles, then an empty dispatch over empty guarded spans.
#[uzu_test]
fn matches_cpu_division() {
    let fixture = KernelFixture::new();
    let kernel = TestExactQuotientVulkanKernel::new(&fixture.context).expect("TestExactQuotient");
    let run = |pairs: &[[u32; 2]]| {
        let expected = pairs.iter().map(|&pair| agree(pair)).collect::<Vec<_>>();
        let output = vec![SENTINEL; pairs.len()];
        let buffers = (fixture.guarded(pairs.as_flattened(), SENTINEL), fixture.guarded(&output, SENTINEL));
        let mut encoding = fixture.encoding();
        // SAFETY: the input holds 2 words and the output 1 word per pair; they do not alias.
        unsafe { kernel.encode(arg(&buffers.0), arg(&buffers.1), pairs.len() as u32, &mut encoding) };
        KernelFixture::complete(encoding);
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe { verify(&buffers, pairs, &expected) };
    };
    let mut pending = Vec::with_capacity(CHUNK);
    visit_families(|pair| {
        pending.push(pair);
        if pending.len() == CHUNK {
            run(&pending);
            pending.clear();
        }
    });
    run(&pending);
    run(&[]);
    fixture.assert_clean();
}

/// Run alone: `cargo test ... exact_quotient_test::throughput -- --ignored --nocapture`; test-only presence flag
/// UZU_QUOTIENT_REVERSE runs the descending round of the 6 cells first: 12 rows. After both host oracles agree,
/// 3 warm-up and 10 timed submissions each use their own guarded pairs and results, each buffer pair checked after
/// completion before the next encode (wall includes it). Host encode: median of the 10 timed. Raw helper timings.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    let kernel = TestExactQuotientVulkanKernel::new(&fixture.context).expect("TestExactQuotient");
    let reverse = std::env::var_os("UZU_QUOTIENT_REVERSE").is_some();
    let cells = ["normal", "edges"].map(|family| [1, 4096, 65536].map(|count| (family, count)));
    let ascending = cells.as_flattened().to_vec();
    let mut rounds = [ascending.clone(), ascending.iter().rev().copied().collect()];
    if reverse {
        rounds.reverse();
    }
    // The witnesses, then every pair of ±0, ±infinity, NaN, a negative subnormal and 1.
    let classes = [0x0000_0000, 0x8000_0000, INFINITY, 0xff80_0000, QUIET_NAN, 0x8000_0001, 0x3f80_0000];
    let edges = WITNESSES
        .iter()
        .map(|&(_, pair, _)| pair)
        .chain(classes.iter().flat_map(|&a| classes.map(|b| [a, b])))
        .collect::<Vec<_>>();
    for (round, cells) in rounds.iter().enumerate() {
        for &(family, count) in cells {
            let operands = (0..count)
                .map(|index| match family {
                    // Numerators in [-2, 2], denominators in [0.5, 2].
                    "normal" => {
                        [((index * 7919) % 4001) as f32 / 1000.0 - 2.0, ((index * 7919) % 3001) as f32 / 2000.0 + 0.5]
                            .map(f32::to_bits)
                    },
                    _ => edges[index % edges.len()],
                })
                .collect::<Vec<_>>();
            let expected = operands.iter().map(|&pair| agree(pair)).collect::<Vec<_>>();
            let output = vec![SENTINEL; count];
            let pairs = (0..13)
                .map(|_| (fixture.guarded(operands.as_flattened(), SENTINEL), fixture.guarded(&output, SENTINEL)))
                .collect::<Vec<_>>();
            let (mut submitted, mut encodes) = (0, Vec::new());
            let (gpu, wall) = fixture.median_times(|encoding| {
                if submitted > 0 {
                    // SAFETY: the submission using this pair has completed and no recording uses it.
                    unsafe { verify(&pairs[submitted - 1], &operands, &expected) };
                }
                let start = Instant::now();
                // SAFETY: each input holds 2 words and its output 1 word per pair; they do not alias.
                unsafe { kernel.encode(arg(&pairs[submitted].0), arg(&pairs[submitted].1), count as u32, encoding) };
                encodes.push(start.elapsed());
                submitted += 1;
            });
            assert_eq!(submitted, pairs.len(), "submissions");
            // SAFETY: median_times has completed every submission.
            unsafe { verify(&pairs[submitted - 1], &operands, &expected) };
            let mut timed = encodes.split_off(3);
            timed.sort();
            let encode = timed[timed.len() / 2];
            eprintln!(
                "MEASURE ExactQuotient reverse {reverse} round {round} family {family} count {count}: median of 10 after 3 warm-up: GPU {gpu:?}, host encode {encode:?}, wall {wall:?} (with the previous check)"
            );
        }
    }
    fixture.assert_clean();
}
