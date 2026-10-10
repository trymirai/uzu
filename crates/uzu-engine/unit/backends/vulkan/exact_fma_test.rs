use std::{ops::Range, sync::Arc, time::Instant};

use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu_engine_macros::uzu_test;

use super::{arg, kernel_fixture::KernelFixture};
use crate::backends::vulkan::{VkBuffer, vk_kernels::TestExactFmaVulkanKernel};

const SENTINEL: u32 = 0xa5a5_a5a5;

/// Triples per GPU submission.
const CHUNK: usize = 1 << 16;

/// The 24 signed classes: ±0, ±min/mid/max subnormal, ±min normal, ±1, ±(1 + ulp), ±(2 - ulp), ±max finite, ±infinity,
/// ±quiet NaN and ±signalling NaN.
const CLASSES: [u32; 24] = [
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x00400000, 0x80400000, 0x007fffff, 0x807fffff, 0x00800000,
    0x80800000, 0x3f800000, 0xbf800000, 0x3f800001, 0xbf800001, 0x3fffffff, 0xbfffffff, 0x7f7fffff, 0xff7fffff,
    0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7fa00000, 0xffa00000,
];

/// Significands 1, 1 + ulp, 1.5 and 2 - ulp of the focused families.
const SIGNIFICANDS: [u32; 4] = [0x3f80_0000, 0x3f80_0001, 0x3fc0_0000, 0x3fff_ffff];

/// Exponent of the product minus exponent of the addend.
const GAPS: [i32; 24] = [-3, -2, -1, 0, 1, 2, 3, 15, 16, 17, 23, 24, 25, 39, 40, 41, 47, 48, 49, 50, 64, 65, 100, 250];

/// `(case, [a, b, c], a * b + c rounded once)`, each derived by hand in P0-REPORT.md; NaN results compare by class.
const WITNESSES: [(&str, [u32; 3], u32); 35] = [
    ("fused cancellation -2^-46", [0x3f80_0001, 0x3f7f_fffe, 0xbf80_0000], 0xa880_0000),
    ("near cancellation 2^-46", [0x3f80_0001, 0x3f80_0001, 0xbf80_0002], 0x2880_0000),
    ("1 + 2^-24 ties to even", [0x3980_0000, 0x3980_0000, 0x3f80_0000], 0x3f80_0000),
    ("odd + 2^-24 ties to even", [0x3980_0000, 0x3980_0000, 0x3f80_0001], 0x3f80_0002),
    ("-(1 + 2^-24) ties to even", [0xb980_0000, 0x3980_0000, 0xbf80_0000], 0xbf80_0000),
    ("-(odd + 2^-24) ties to even", [0xb980_0000, 0x3980_0000, 0xbf80_0001], 0xbf80_0002),
    ("2^-70 below an odd midpoint", [0x3980_0001, 0x397f_fffe, 0x3f80_0001], 0x3f80_0001),
    ("2^-70 inside a negative odd midpoint", [0xb980_0001, 0x397f_fffe, 0xbf80_0001], 0xbf80_0001),
    ("product midpoint plus 2^-60", [0x3f80_0800, 0x3f80_0800, 0x2180_0000], 0x3f80_1001),
    ("product midpoint minus 2^-60", [0x3f80_0800, 0x3f80_0800, 0xa180_0000], 0x3f80_1000),
    ("product midpoint plus +0", [0x3f80_0800, 0x3f80_0800, 0x0000_0000], 0x3f80_1000),
    ("2 - 2^-24 carries to 2", [0x3980_0000, 0x3980_0000, 0x3fff_ffff], 0x4000_0000),
    ("2^79 below the overflow midpoint", [0x7300_0000, 0x3f7f_ffff, 0x7f7f_ffff], 0x7f7f_ffff),
    ("at the overflow midpoint", [0x7300_0000, 0x3f80_0000, 0x7f7f_ffff], 0x7f80_0000),
    ("2^80 above the overflow midpoint", [0x7300_0000, 0x3f80_0001, 0x7f7f_ffff], 0x7f80_0000),
    ("2^57 below the overflow midpoint", [0x7300_0001, 0x3f7f_fffe, 0x7f7f_ffff], 0x7f7f_ffff),
    ("2^57 inside the negative overflow midpoint", [0xf300_0001, 0x3f7f_fffe, 0xff7f_ffff], 0xff7f_ffff),
    ("product overflow rescued by c", [0x5f80_0001, 0x5f80_0000, 0xff7f_ffff], 0x7440_0000),
    ("max subnormal + 2^-150 ties to min normal", [0x1a00_0000, 0x1a00_0000, 0x007f_ffff], 0x0080_0000),
    ("below the min normal midpoint", [0x1a00_0000, 0x19ff_fffe, 0x007f_ffff], 0x007f_ffff),
    ("min normal - 2^-149 is max subnormal", [0x1a00_0000, 0x9a80_0000, 0x0080_0000], 0x007f_ffff),
    ("2^-150 ties to +0", [0x1a00_0000, 0x1a00_0000, 0x0000_0000], 0x0000_0000),
    ("-2^-150 ties to -0", [0x9a00_0000, 0x1a00_0000, 0x0000_0000], 0x8000_0000),
    ("above 2^-150 rounds to 2^-149", [0x1a00_0000, 0x1a00_0001, 0x0000_0000], 0x0000_0001),
    ("3 * 2^-150 ties to 2^-148", [0x1a00_0000, 0x1a00_0000, 0x0000_0001], 0x0000_0002),
    ("tiny positive product + -0 is +0", [0x0000_0001, 0x0000_0001, 0x8000_0000], 0x0000_0000),
    ("tiny negative product + +0 is -0", [0x8000_0001, 0x0000_0001, 0x0000_0000], 0x8000_0000),
    ("exact cancellation is +0", [0x3f80_0000, 0x3f80_0000, 0xbf80_0000], 0x0000_0000),
    ("negative exact cancellation is +0", [0xbf80_0000, 0x3f80_0000, 0x3f80_0000], 0x0000_0000),
    ("+0 product + -0 is +0", [0x0000_0000, 0x3f80_0000, 0x8000_0000], 0x0000_0000),
    ("-0 product + -0 is -0", [0x8000_0000, 0x3f80_0000, 0x8000_0000], 0x8000_0000),
    ("-0 product + +0 is +0", [0x0000_0000, 0xbf80_0000, 0x0000_0000], 0x0000_0000),
    ("zero product keeps a subnormal c", [0x0000_0000, 0x7f7f_ffff, 0x8000_0001], 0x8000_0001),
    ("zero times infinity is NaN", [0x0000_0000, 0x7f80_0000, 0x3f80_0000], 0x7fc0_0000),
    ("infinite cancellation is NaN", [0x7f80_0000, 0x3f80_0000, 0xff80_0000], 0x7fc0_0000),
];

/// a * b + c rounded once to nearest even, independently of `mul_add`: the FP64 product of two FP32 values is exact
/// (at most 48 significand bits, magnitudes in [2^-298, 2^256]) and Knuth's TwoSum gives the exact sum as
/// `sum + residual`. Every FP32 rounding boundary is an FP64 value, so rounding to nearest keeps `sum` on the side of
/// each boundary the exact sum is on: the residual decides only when `sum` is itself an FP32 midpoint, the overflow
/// midpoint 2^128 - 2^103 between max finite and 2^128 (which stands for infinity) included.
fn fma_oracle(
    a: f32,
    b: f32,
    c: f32,
) -> f32 {
    let (product, addend) = (f64::from(a) * f64::from(b), f64::from(c));
    let sum = product + addend;
    if !sum.is_finite() {
        return sum as f32;
    }
    let addend_part = sum - product;
    let residual = (product - (sum - addend_part)) + (addend - addend_part);
    let rounded = sum as f32;
    let exact = if rounded.is_infinite() {
        f64::from_bits(0x47f0_0000_0000_0000).copysign(sum)
    } else {
        f64::from(rounded)
    };
    if residual == 0.0 || exact == sum {
        return rounded;
    }
    let neighbour = if exact < sum {
        rounded.next_up()
    } else {
        rounded.next_down()
    };
    let midpoint = (exact + f64::from(neighbour)) / 2.0 == sum;
    if midpoint && (residual > 0.0) == (f64::from(neighbour) > sum) {
        neighbour
    } else {
        rounded
    }
}

/// Asserts `mul_add` (expected) and the oracle (actual) agree on `[a, b, c]` bits and returns the result.
fn agree([a, b, c]: [u32; 3]) -> f32 {
    let (fused, oracle) = (
        f32::from_bits(a).mul_add(f32::from_bits(b), f32::from_bits(c)),
        fma_oracle(f32::from_bits(a), f32::from_bits(b), f32::from_bits(c)),
    );
    KernelFixture::assert_bits(&[fused], &[oracle], &format!("[{a:#010x}, {b:#010x}, {c:#010x}] mul_add vs oracle"));
    fused
}

/// 2^exponent for a normal exponent.
fn power_of_two(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

/// Every triple of the 24 classes.
fn visit_classes(mut visit: impl FnMut([u32; 3])) {
    for a in CLASSES {
        for b in CLASSES {
            for c in CLASSES {
                visit([a, b, c]);
            }
        }
    }
}

/// 2^20 uniform bit triples, then 2^20 products near the subnormal edge, the min normal, one and max finite, cancelled
/// by c within two bits of -a * b.
fn visit_random(mut visit: impl FnMut([u32; 3])) {
    let mut rng = SmallRng::seed_from_u64(0x0fa5_7f3a);
    for _ in 0..1 << 20 {
        visit([0; 3].map(|_| rng.random_range(..=u32::MAX)));
    }
    for _ in 0..1 << 20 {
        let target = [0, 1, 127, 254][rng.random_range(0..4)];
        let a_exponent = rng.random_range(1..=254i32);
        let b_exponent = (target + 127 + rng.random_range(-1..=1) - a_exponent).clamp(1, 254);
        let [a, b] = [a_exponent, b_exponent]
            .map(|exponent| rng.random_range(0..2u32) << 31 | (exponent as u32) << 23 | rng.random_range(..1u32 << 23));
        let product = f32::from_bits(a) * f32::from_bits(b);
        visit([a, b, (-product).to_bits().wrapping_add(rng.random_range(0..5)).wrapping_sub(2)]);
    }
}

/// Every gap with every significand triple and both addend signs, then product rounding alone (c = ±0) across the
/// subnormal, min normal and overflow edges.
fn visit_gaps(mut visit: impl FnMut([u32; 3])) {
    for gap in GAPS {
        let (addend_exponent, product_exponent) = (-(gap / 2), gap - gap / 2);
        for a in SIGNIFICANDS {
            for b in SIGNIFICANDS {
                for c in SIGNIFICANDS {
                    for sign in [0, 0x8000_0000] {
                        let a = (power_of_two(product_exponent) * f32::from_bits(a)).to_bits();
                        let c = (power_of_two(addend_exponent) * f32::from_bits(c)).to_bits() | sign;
                        visit([a, b, c]);
                    }
                }
            }
        }
    }
    for exponent_sum in [-152, -151, -150, -149, -148, -147, -127, -126, -125, 124, 125, 126, 127, 128] {
        for a in SIGNIFICANDS {
            for b in SIGNIFICANDS {
                let a = (power_of_two(exponent_sum / 2) * f32::from_bits(a)).to_bits();
                let b = (power_of_two(exponent_sum - exponent_sum / 2) * f32::from_bits(b)).to_bits();
                visit([a, b, 0]);
                visit([a, b | 0x8000_0000, 0x8000_0000]);
            }
        }
    }
}

/// Asserts a completed TestExactFma dispatch kept its guarded triples and wrote `expected` between untouched guards.
///
/// # Safety
/// Every command buffer using the buffers has completed.
unsafe fn verify(
    (input, output): &((Arc<VkBuffer>, Range<u64>), (Arc<VkBuffer>, Range<u64>)),
    triples: &[[u32; 3]],
    expected: &[f32],
) {
    // SAFETY: the caller guarantees completion.
    let actual = unsafe {
        KernelFixture::assert_unchanged(input, SENTINEL, triples.as_flattened(), "triples");
        KernelFixture::read_guarded::<u32>(output, SENTINEL)
    };
    KernelFixture::assert_bits(expected, &actual.into_iter().map(f32::from_bits).collect::<Vec<_>>(), "TestExactFma");
}

#[uzu_test]
fn every_class_triple_agrees() {
    visit_classes(|triple| {
        agree(triple);
    });
}

#[uzu_test]
fn frozen_witnesses_match_both_oracles() {
    for (case, inputs, expected) in WITNESSES {
        KernelFixture::assert_bits(&[f32::from_bits(expected)], &[agree(inputs)], case);
    }
}

#[uzu_test]
fn random_triples_agree() {
    visit_random(|triple| {
        agree(triple);
    });
}

#[uzu_test]
fn exponent_gap_families_agree() {
    visit_gaps(|triple| {
        agree(triple);
    });
}

/// Every host family through TestExactFma in submissions of at most CHUNK triples, each first checked by both host
/// oracles, then an empty dispatch over empty guarded spans.
#[uzu_test]
fn matches_cpu_fused_rounding() {
    let fixture = KernelFixture::new();
    let kernel = TestExactFmaVulkanKernel::new(&fixture.context).expect("TestExactFma");
    let run = |triples: &[[u32; 3]]| {
        let expected = triples.iter().map(|&triple| agree(triple)).collect::<Vec<_>>();
        let output = vec![SENTINEL; triples.len()];
        let pair = (fixture.guarded(triples.as_flattened(), SENTINEL), fixture.guarded(&output, SENTINEL));
        let mut encoding = fixture.encoding();
        // SAFETY: the input holds 3 words and the output 1 word per triple; they do not alias.
        unsafe { kernel.encode(arg(&pair.0), arg(&pair.1), triples.len() as u32, &mut encoding) };
        KernelFixture::complete(encoding);
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe { verify(&pair, triples, &expected) };
    };
    let mut pending = Vec::with_capacity(CHUNK);
    let mut push = |triple: [u32; 3]| {
        pending.push(triple);
        if pending.len() == CHUNK {
            run(&pending);
            pending.clear();
        }
    };
    visit_classes(&mut push);
    WITNESSES.iter().for_each(|&(_, triple, _)| push(triple));
    visit_gaps(&mut push);
    visit_random(&mut push);
    run(&pending);
    run(&[]);
    fixture.assert_clean();
}

/// Run alone: `cargo test ... exact_fma_test::throughput -- --ignored --nocapture`; test-only presence flag
/// UZU_FMA_REVERSE runs the descending round of the 6 cells first: 12 rows. After both host oracles agree, 3 warm-up and
/// 10 timed submissions each use their own guarded triples and results, each pair checked after completion before the
/// next encode (wall includes it). Host encode: median of the 10 timed. Raw helper timings.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    let kernel = TestExactFmaVulkanKernel::new(&fixture.context).expect("TestExactFma");
    let reverse = std::env::var_os("UZU_FMA_REVERSE").is_some();
    let cells = ["normal", "edges"].map(|family| [1, 4096, 65536].map(|count| (family, count)));
    let ascending = cells.as_flattened().to_vec();
    let mut rounds = [ascending.clone(), ascending.iter().rev().copied().collect()];
    if reverse {
        rounds.reverse();
    }
    let finite = |index: usize| (((index * 7919) % 4001) as f32 / 1000.0 - 2.0).to_bits();
    for (round, cells) in rounds.iter().enumerate() {
        for &(family, count) in cells {
            let triples = (0..count)
                .map(|index| match family {
                    "normal" => [0, 1, 2].map(|operand| finite(3 * index + operand)),
                    _ => WITNESSES[index % WITNESSES.len()].1,
                })
                .collect::<Vec<_>>();
            let expected = triples.iter().map(|&triple| agree(triple)).collect::<Vec<_>>();
            let output = vec![SENTINEL; count];
            let pairs = (0..13)
                .map(|_| (fixture.guarded(triples.as_flattened(), SENTINEL), fixture.guarded(&output, SENTINEL)))
                .collect::<Vec<_>>();
            let (mut submitted, mut encodes) = (0, Vec::new());
            let (gpu, wall) = fixture.median_times(|encoding| {
                if submitted > 0 {
                    // SAFETY: the submission using this pair has completed and no recording uses it.
                    unsafe { verify(&pairs[submitted - 1], &triples, &expected) };
                }
                let start = Instant::now();
                // SAFETY: each input holds 3 words and its output 1 word per triple; they do not alias.
                unsafe { kernel.encode(arg(&pairs[submitted].0), arg(&pairs[submitted].1), count as u32, encoding) };
                encodes.push(start.elapsed());
                submitted += 1;
            });
            assert_eq!(submitted, pairs.len(), "submissions");
            // SAFETY: median_times has completed every submission.
            unsafe { verify(&pairs[submitted - 1], &triples, &expected) };
            let mut timed = encodes.split_off(3);
            timed.sort();
            let encode = timed[timed.len() / 2];
            eprintln!(
                "MEASURE ExactFma reverse {reverse} round {round} family {family} count {count}: median of 10 after 3 warm-up: GPU {gpu:?}, host encode {encode:?}, wall {wall:?} (with the previous check)"
            );
        }
    }
    fixture.assert_clean();
}
