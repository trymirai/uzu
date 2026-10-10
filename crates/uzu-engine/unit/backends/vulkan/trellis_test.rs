use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{
    cpu_buffer, cpu_submissions, fma_oracle, kernel_fixture::KernelFixture, quantized, signs, transform_oracle, values,
};
use crate::{
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::trellis::COLUMN_GROUP_COUNT,
            kernel::{TrellisTransformKernel, matmul::Int8CodeLayout, mixing_dimension},
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_context},
};

const DIMENSIONS: [u32; 6] = [4096, 5120, 6144, 6656, 17408, 19968];

/// The CPU's FP32 reciprocal of sqrt(block) for every Hadamard block of ActivationTransform and Trellis.
const NORMALIZATIONS: [(usize, u32); 5] =
    [(32, 0x3e35_04f3), (512, 0x3d35_04f3), (1024, 0x3d00_0000), (2048, 0x3cb5_04f3), (4096, 0x3c80_0000)];

/// Root's scalar normalization witnesses (ROOT-NORMALIZATION-WITNESSES.json): (block, factor, normalization one ULP off,
/// correct BF16, wrong BF16, correct scale, wrong scale) for input 1 times the factor at h = 0 and mixing[0] = 1.
const NORMALIZATION_WITNESSES: [(usize, u32, u32, u16, u16, u32, u32); 8] = [
    (512, 1065607195, 1026884852, 15674, 15675, 968586990, 968653042),
    (512, 1065653536, 1026884850, 15676, 15675, 968719094, 968653042),
    (1024, 1107329024, 1023410177, 16256, 16257, 1006699012, 1006765064),
    (1024, 1107394560, 1023410175, 16258, 16257, 1006831116, 1006765064),
    (2048, 1065607195, 1018496244, 15546, 15547, 960198382, 960264434),
    (2048, 1065653536, 1018496242, 15548, 15547, 960330486, 960264434),
    (4096, 1115717632, 1015021569, 16256, 16257, 1006699012, 1006765064),
    (4096, 1115783168, 1015021567, 16258, 16257, 1006831116, 1006765064),
];

/// BF16 classes: quiet and signalling NaN, ±infinity, ±0, the smallest and largest subnormal, the smallest normal, the
/// largest finite value and ±1.
const SPECIALS: [u16; 12] =
    [0x7fc1, 0x7f81, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x007f, 0x0080, 0x7f7f, 0x3f80, 0xbf80];

const ONE: u16 = 0x3f80;
const UNIT: u32 = 0x3f80_0000;

/// `batch` rows of zero inputs, factors 1 and zero mixing, with the given `(index, bits)` entries of each.
fn case(
    dimension: u32,
    batch: usize,
    input: &[(usize, u16)],
    factors: &[(usize, u32)],
    mixing: &[(usize, u32)],
) -> (Vec<bf16>, Vec<f32>, Vec<f32>) {
    let (columns, m) = (dimension as usize, mixing_dimension(dimension) as usize);
    let mut data = (vec![bf16::ZERO; batch * columns], vec![1.0f32; columns], vec![0.0f32; m * m]);
    input.iter().for_each(|&(index, bits)| data.0[index] = bf16::from_bits(bits));
    factors.iter().for_each(|&(index, bits)| data.1[index] = f32::from_bits(bits));
    mixing.iter().for_each(|&(index, bits)| data.2[index] = f32::from_bits(bits));
    data
}

/// The CPU kernel through the shared trait: codes, column-group sums and scales, after asserting the inputs kept their
/// bits.
fn cpu(
    dimension: u32,
    (input, factors, mixing): &(Vec<bf16>, Vec<f32>, Vec<f32>),
) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let batch = input.len() / dimension as usize;
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::TrellisTransformKernel::new(&context, dimension)
        .expect("CPU TrellisTransform");
    let (input_buffer, factor_buffer, mixing_buffer) =
        (cpu_buffer(&context, input), cpu_buffer(&context, factors), cpu_buffer(&context, mixing));
    let mut codes = cpu_buffer(&context, &vec![0x5a_i8; input.len()]);
    let mut sums = cpu_buffer(&context, &vec![-7.0f32; COLUMN_GROUP_COUNT as usize * batch]);
    let mut scales = cpu_buffer(&context, &vec![-7.0f32; batch]);
    cpu_submissions(&context, 1, |command_buffer| {
        let (input, factors, mixing) = (&input_buffer, &factor_buffer, &mixing_buffer);
        kernel.encode(input, factors, mixing, &mut codes, &mut sums, &mut scales, batch as u32, command_buffer);
    });
    let words = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let halves = |values: &[bf16]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    assert_eq!(halves(&buffer_to_vec::<Cpu, bf16>(&input_buffer)), halves(input), "input changed");
    assert_eq!(words(&buffer_to_vec::<Cpu, f32>(&factor_buffer)), words(factors), "factors changed");
    assert_eq!(words(&buffer_to_vec::<Cpu, f32>(&mixing_buffer)), words(mixing), "mixing changed");
    (buffer_to_vec::<Cpu, i8>(&codes), buffer_to_vec::<Cpu, f32>(&sums), buffer_to_vec::<Cpu, f32>(&scales))
}

/// The independent exact oracle. Output column o of each row over h < H = D / M folds the separately rounded product of
/// the widened input and the factor with mixing[o M + i] through `fma_oracle`, i ascending from +0; then the canonical
/// staged transform of block H, BF16 rounding, the canonical CPU quantization of whole rows and the sums by column mod 4.
fn oracle(
    dimension: u32,
    (input, factors, mixing): &(Vec<bf16>, Vec<f32>, Vec<f32>),
) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let (columns, m) = (dimension as usize, mixing_dimension(dimension) as usize);
    let block = columns / m;
    let mut rotated = vec![0.0f32; input.len()];
    for (row, values) in rotated.chunks_exact_mut(columns).zip(input.chunks_exact(columns)) {
        for o in 0..m {
            let mixed = |h: usize| {
                (0..m).fold(0.0f32, |acc, i| {
                    fma_oracle(values[h * m + i].to_f32() * factors[h * m + i], mixing[o * m + i], acc)
                })
            };
            let points = (0..block).map(|h| f64::from(mixed(h))).map(|y| ((y, y), y)).collect::<Vec<_>>();
            for (h, (_, z)) in transform_oracle(&points, &vec![1; block], true, block).into_iter().enumerate() {
                row[h * m + o] = bf16::from_f32(z as f32).to_f32();
            }
        }
    }
    let (codes, scales, _) = quantized(&rotated, columns, (columns, None, Int8CodeLayout::Sequential));
    let groups = COLUMN_GROUP_COUNT as usize;
    let sums = codes
        .chunks_exact(columns)
        .flat_map(|row| {
            let mut sums = vec![0i32; groups];
            row.iter().enumerate().for_each(|(column, &code)| sums[column % groups] += i32::from(code));
            sums.into_iter().map(|sum| sum as f32)
        })
        .collect();
    (codes, sums, scales)
}

/// Asserts the CPU kernel and the oracle agree bit for bit and returns the CPU outputs.
fn agree(
    case: &str,
    dimension: u32,
    data: &(Vec<bf16>, Vec<f32>, Vec<f32>),
) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let (actual, expected) = (cpu(dimension, data), oracle(dimension, data));
    let code = actual.0.iter().zip(&expected.0).position(|(actual, expected)| actual != expected);
    assert!(actual.0.len() == expected.0.len() && code.is_none(), "{case}: CPU code {code:?} differs from the oracle");
    KernelFixture::assert_bits(&expected.1, &actual.1, &format!("{case}: sums"));
    KernelFixture::assert_bits(&expected.2, &actual.2, &format!("{case}: scales"));
    actual
}

/// A frozen witness: the CPU and the oracle agree, then each row's scale bits and column-group sums, and the code of
/// column h M + o of row r equal `scales`, `sums` and `code(r, h, o)`.
fn witness(
    case: &str,
    dimension: u32,
    data: &(Vec<bf16>, Vec<f32>, Vec<f32>),
    scales: &[u32],
    sums: &[[i32; 4]],
    code: impl Fn(usize, usize, usize) -> i8,
) {
    let (codes, actual_sums, actual_scales) = agree(case, dimension, data);
    let (columns, m) = (dimension as usize, mixing_dimension(dimension) as usize);
    let wrong = (0..codes.len()).find(|&i| codes[i] != code(i / columns, i % columns / m, i % m));
    assert!(wrong.is_none(), "{case}: code {wrong:?} differs from the frozen pattern");
    let sums = sums.as_flattened().iter().map(|&sum| sum as f32).collect::<Vec<_>>();
    KernelFixture::assert_bits(&sums, &actual_sums, &format!("{case}: frozen sums"));
    let scales = scales.iter().map(|&bits| f32::from_bits(bits)).collect::<Vec<_>>();
    KernelFixture::assert_bits(&scales, &actual_scales, &format!("{case}: frozen scales"));
}

#[uzu_test]
fn normalization_constants() {
    for (block, bits) in NORMALIZATIONS {
        assert_eq!((1.0 / (block as f32).sqrt()).to_bits(), bits, "block {block}");
    }
}

/// Every dimension at one and three rows (the middle one all zero), with sign and general finite factors.
#[uzu_test]
fn finite_rows_match_oracle() {
    for dimension in DIMENSIONS {
        let (columns, m) = (dimension as usize, mixing_dimension(dimension) as usize);
        for batch in [1, 3] {
            let sign_factors = signs(columns, batch).into_iter().map(|sign| sign as f32).collect::<Vec<_>>();
            for factors in [sign_factors, values::<f32>(columns, batch + 1)] {
                let mut input = values::<bf16>(batch * columns, batch + 2);
                if batch == 3 {
                    input[columns..2 * columns].fill(bf16::ZERO);
                }
                let data = (input, factors, values::<f32>(m * m, batch + 3));
                agree(&format!("D {dimension} B {batch}"), dimension, &data);
            }
        }
    }
}

/// Every dimension with one homogeneous row per BF16 class, unit factors and identity mixing; and the future native
/// flush-to-zero detector: minimum subnormal inputs times 32 mix to the subnormal 2^-128, and only the transform's
/// h = 0 sum H 2^-128 reaches the codes, at Root's subnormal scale (ROOT-FTZ-SCALE-WITNESSES.json).
#[uzu_test]
fn class_rows_match_oracle() {
    let ftz_scales = [0x0010_2041, 0x0008_1020, 0x000b_66ce, 0x0005_b367, 0x0008_1020, 0x0005_b367];
    for (dimension, ftz_scale) in DIMENSIONS.into_iter().zip(ftz_scales) {
        let (columns, m) = (dimension as usize, mixing_dimension(dimension) as usize);
        let identity = (0..m * m)
            .map(|index| {
                if index.is_multiple_of(m + 1) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f32>>();
        let input = SPECIALS.iter().flat_map(|&bits| vec![bf16::from_bits(bits); columns]).collect();
        agree(&format!("classes D {dimension}"), dimension, &(input, vec![1.0; columns], identity.clone()));
        let data = (vec![bf16::from_bits(0x0001); columns], vec![32.0; columns], identity);
        let (codes, sums, scales) = agree(&format!("subnormal mixing D {dimension}"), dimension, &data);
        assert!(codes.iter().enumerate().all(|(column, &code)| code
            == if column < m {
                127
            } else {
                0
            }));
        let expected = (0..4).map(|group| (127 * (0..m).filter(|o| o % 4 == group).count()) as f32).collect::<Vec<_>>();
        KernelFixture::assert_bits(&expected, &sums, &format!("subnormal mixing D {dimension} sums"));
        let scale = [f32::from_bits(ftz_scale)];
        KernelFixture::assert_bits(&scale, &scales, &format!("subnormal mixing D {dimension} scale"));
    }
}

/// Hand-derived (H0-REPORT.md) or Root-derived literal witnesses, each checked against both the CPU and the oracle.
#[uzu_test]
fn frozen_witnesses() {
    let column0 = |value: i8| {
        move |_: usize, _: usize, o: usize| {
            if o == 0 {
                value
            } else {
                0
            }
        }
    };
    for (block, factor, wrong_norm, correct, wrong, scale, wrong_scale) in NORMALIZATION_WITNESSES {
        let (factor_value, norm) = (f32::from_bits(factor), 1.0 / (block as f32).sqrt());
        let rounded =
            [factor_value * norm, factor_value * f32::from_bits(wrong_norm)].map(|y| bf16::from_f32(y).to_bits());
        assert_eq!(rounded, [correct, wrong], "WN {block} {factor:#x} BF16");
        assert_eq!((bf16::from_bits(wrong).to_f32() / 127.0).to_bits(), wrong_scale, "WN {block} {factor:#x} scale");
        for dimension in DIMENSIONS.into_iter().filter(|&d| d as usize / mixing_dimension(d) as usize == block) {
            let data = case(dimension, 1, &[(0, ONE)], &[(0, factor)], &[(0, UNIT)]);
            let sums = [127 * block as i32 / 4; 4];
            witness(
                &format!("WN {block} {factor:#x} D {dimension}"),
                dimension,
                &data,
                &[scale],
                &[sums],
                column0(127),
            );
        }
    }
    // WF: (1 + 2^-23)(1 - 2^-23) - 1 = -2^-46 fused at h = 0, column 0; separate rounding gives zero rows and scale 1.
    for (dimension, scale, sum) in [
        (5120, 0x2281_0204, -32512),
        (17408, 0x2281_0204, -32512),
        (6144, 0x2236_6cda, -65024),
        (6656, 0x22b6_6cda, -16256),
        (19968, 0x22b6_6cda, -16256),
    ] {
        let data = case(
            dimension,
            1,
            &[(0, ONE), (1, ONE)],
            &[(0, 0xbf80_0000), (1, 0x3f80_0001)],
            &[(0, UNIT), (1, 0x3f7f_fffe)],
        );
        witness(&format!("WF D {dimension}"), dimension, &data, &[scale], &[[sum; 4]], column0(-127));
    }
    // WO: -1, 2^-24, 1 ascending give 2^-24; descending gives +0.
    for dimension in [5120, 17408] {
        let data = case(dimension, 1, &[(0, 0xbf80), (1, 0x3380), (2, ONE)], &[], &[(0, UNIT), (1, UNIT), (2, UNIT)]);
        witness(&format!("WO D {dimension}"), dimension, &data, &[0x2d81_0204], &[[32512; 4]], column0(127));
    }
    // WB: 64.75 / 64 = 0x3f818000 is a BF16 tie rounding to 0x3f82; unrounded it gives another scale.
    let data = case(4096, 1, &[(0, ONE)], &[(0, 0x4281_8000)], &[(0, UNIT)]);
    witness("WB D 4096", 4096, &data, &[0x3c03_060c], &[[130048; 4]], column0(127));
    // WM: (1 + 2^-23)(1 + 2^-8 - 2^-23) rounds in FP32 to the BF16 midpoint 1 + 2^-8, which ties to 0x3c80 after the
    // normalization 2^-6; the exact product lies just above it, so an upward-rounded product would give 0x3c81.
    let data = case(4096, 1, &[(0, ONE)], &[(0, 0x3f80_0001)], &[(0, 0x3f80_7fff)]);
    witness("WM D 4096", 4096, &data, &[0x3901_0204], &[[130048; 4]], column0(127));
    // WH: factor A at h = 0, 2^-24 at 2^a and 2^-23 at 2^b. FP32 stages round ties to even: in ascending order,
    // A ± 2^-24 keeps the even A = 1 + 2^-8 (H 1024/4096), so only bit b decides, but moves the odd A = 0x3f83e01b
    // (H 512/2048) one ULP, so 127 needs bits a and b both 0. Swapping the two stages exchanges these rules.
    for (dimension, a, b, factor, mask, scale, sums) in [
        (4096, 0, 1, 0x3f80_8000, 0x2, 0x3902_0408, [130048, 130048, 129024, 129024]),
        (4096, 4, 5, 0x3f80_8000, 0x20, 0x3902_0408, [129536; 4]),
        (5120, 0, 1, 0x3f80_8000, 0x2, 0x3982_0408, [32512, 32512, 32256, 32256]),
        (5120, 3, 4, 0x3f80_8000, 0x10, 0x3982_0408, [32384; 4]),
        (5120, 5, 6, 0x3f80_8000, 0x40, 0x3982_0408, [32384; 4]),
        (6144, 3, 4, 0x3f83_e01b, 0x18, 0x393c_78f2, [64640; 4]),
        (6144, 6, 7, 0x3f83_e01b, 0xc0, 0x393c_78f2, [64640; 4]),
        (6656, 3, 4, 0x3f83_e01b, 0x18, 0x39bc_78f2, [16160; 4]),
        (6656, 6, 7, 0x3f83_e01b, 0xc0, 0x39bc_78f2, [16160; 4]),
        (19968, 3, 4, 0x3f83_e01b, 0x18, 0x39bc_78f2, [16160; 4]),
        (19968, 6, 7, 0x3f83_e01b, 0xc0, 0x39bc_78f2, [16160; 4]),
    ] {
        let m = mixing_dimension(dimension) as usize;
        let data = case(dimension, 1, &[(0, ONE), (m << a, 0x3380), (m << b, 0x3400)], &[(0, factor)], &[(0, UNIT)]);
        let code = |_: usize, h: usize, o: usize| match o {
            0 if h & mask == 0 => 127,
            0 => 126,
            _ => 0,
        };
        witness(&format!("WH D {dimension} bits {a} {b}"), dimension, &data, &[scale], &[sums], code);
    }
    // WP: 1 at h = 0 and 1/4 at h = 16, distinct across the in-place pass bit 4.
    let data = case(4096, 1, &[(0, ONE), (16, 0x3e80)], &[], &[(0, UNIT)]);
    let code = |_: usize, h: usize, _: usize| {
        if h >> 4 & 1 == 0 {
            127
        } else {
            76
        }
    };
    witness("WP D 4096", 4096, &data, &[0x3921_4285], &[[103936; 4]], code);
    // WT and WS: row 0 ties away from zero at scale 2^-5; row 1 places ±127 at column 5h + 1, summed by column mod 4.
    let mixing = [(0, 0x42fe_0000), (5, 0x4020_0000), (10, 0x3f00_0000), (15, 0xc020_0000), (6, 0x457e_0000)];
    let data = case(5120, 2, &[(0, ONE), (5120 + 6, ONE)], &[], &mixing);
    let code = |row: usize, h: usize, o: usize| match (row, o) {
        (0, _) => [127, 3, 1, -3, 0][o],
        (_, 1) if h.is_multiple_of(2) => 127,
        (_, 1) => -127,
        _ => 0,
    };
    let sums = [[32768; 4], [-32512, 32512, -32512, 32512]];
    witness("WT WS D 5120", 5120, &data, &[0x3d00_0000, 0x3f80_0000], &sums, code);
    // Mixed NaN: finite impulse column 0, NaN column 1, zero elsewhere; NaN not masked from the maximum gives scale 1.
    for dimension in [5120, 17408] {
        let m = mixing_dimension(dimension) as usize;
        let data = case(dimension, 1, &[(0, ONE)], &[], &[(0, UNIT), (m, 0x7fc0_0000)]);
        witness(&format!("mixed NaN D {dimension}"), dimension, &data, &[0x3981_0204], &[[32512; 4]], column0(127));
    }
}
