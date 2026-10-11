use std::{collections::BTreeMap, ops::Range, sync::Arc};

use uzu_engine_macros::uzu_test;

use super::{MatmulCase as Case, kernel_fixture::KernelFixture};
use crate::{
    backends::{
        common::{
            gpu_types::{
                QuantizationMethod::{self, ScaleBias, ScaleSymmetric, ScaleZeroPoint},
                QuantizationMode,
            },
            kernel::matmul::{
                Int8CodeLayout,
                QuantParamsLayout::{self, GroupOutput, OutputGroup},
            },
        },
        vulkan::{Error, QuantizedGemmVulkanKernel, QuantizedGemvVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::matmul::qwen3_layer_shapes,
};

/// The 72 decode configurations (bits, group size, method, metadata layout, stored signed codes), 2 bits x 3 groups x 3 methods x 2 layouts x 2 signed, then 4 of the non-power group
/// 48, which every positive group size must support.
fn configurations() -> Vec<(u32, u32, QuantizationMethod, QuantParamsLayout, bool)> {
    let mut configurations = itertools::iproduct!(
        [4, 8],
        [32, 64, 128],
        [ScaleBias, ScaleZeroPoint, ScaleSymmetric],
        [OutputGroup, GroupOutput],
        [false, true]
    )
    .collect::<Vec<_>>();
    assert_eq!(configurations.len(), 72);
    configurations.extend([
        (4, 48, ScaleZeroPoint, OutputGroup, true),
        (8, 48, ScaleBias, GroupOutput, false),
        (4, 48, ScaleSymmetric, GroupOutput, false),
        (8, 48, ScaleZeroPoint, GroupOutput, true),
    ]);
    configurations
}

fn quantized(
    case: Case,
    (bits, group, method, layout, signed): (u32, u32, QuantizationMethod, QuantParamsLayout, bool),
    seed: u64,
) -> Case {
    case.quantize(bits, group, method, layout, signed, seed)
}

/// The case's `check` through QuantizedGemv, or QuantizedGemm, which without a soft cap also stores the CPU's bits as
/// its K fold is the CPU's order; with prepared A through A8QuantizedGemv or A8QuantizedGemm.
pub fn check(
    fixture: &KernelFixture,
    label: &str,
    case: &Case,
    gemm: bool,
    totals: &mut BTreeMap<String, [f64; 3]>,
) -> (Vec<f32>, Vec<f32>) {
    let run = match (gemm, case.prepared().is_some()) {
        (false, false) => Case::quantized_gemv,
        (true, false) => Case::quantized_gemm,
        (false, true) => Case::a8_quantized_gemv,
        (true, true) => Case::a8_quantized_gemm,
    };
    let (cpu, vulkan) = case.check(fixture, label, run, totals);
    if gemm && case.soft_cap.is_none() {
        KernelFixture::assert_bits(&cpu, &vulkan, &format!("{label} {} CPU bits", case.label()));
    }
    (cpu, vulkan)
}

/// Every case through `check`; returns how many ran.
pub fn check_all(
    label: &str,
    gemm: bool,
    cases: impl IntoIterator<Item = Case>,
) -> usize {
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    let (mut count, mut a8) = (0, "");
    for case in cases {
        check(&fixture, label, &case, gemm, &mut totals);
        count += 1;
        if case.prepared().is_some() {
            a8 = "A8";
        }
    }
    let kernel = if gemm {
        "QuantizedGemm"
    } else {
        "QuantizedGemv"
    };
    Case::report(&format!("{a8}{kernel}"), &totals);
    fixture.assert_clean();
    count
}

/// The host decode `b` holds is the CPU MatmulKernel's, bit for bit: the CPU against an identity A in F32 stores each
/// finite decoded weight w(column, row) as D[row, column], every other product an exact zero, for every configuration
/// in F32 and BF16 metadata over 97 x 33 (k x rows). A decoded zero is compared by value, as the CPU's fold from +0
/// stores +0 for -0. Finite weights only: zero times an infinity or NaN would reach other outputs.
#[uzu_test]
fn host_decode_matches_cpu() {
    let (k, rows) = (97, 33);
    let mut count = 0;
    for ((index, configuration), metadata) in
        itertools::iproduct!(configurations().into_iter().enumerate(), [DataType::F32, DataType::BF16])
    {
        let mut case = quantized(
            Case::new([metadata, DataType::F32, DataType::F32], k, rows, k, 0, 1),
            configuration,
            index as u64,
        );
        case.a = (0..k * k)
            .map(|position| {
                if position / k == position % k {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let cpu = case.cpu(1, false);
        for (position, &stored) in cpu.iter().enumerate() {
            let (row, column) = (position / rows as usize, position % rows as usize);
            let decoded = case.b[column * k as usize + row];
            assert!(decoded.is_finite(), "{}: decoded {decoded:e}", case.label());
            assert!(
                stored.to_bits() == decoded.to_bits() || (decoded == 0.0 && stored == 0.0),
                "{}: weight ({column}, {row}) CPU {stored:e}, host {decoded:e}",
                case.label()
            );
        }
        count += 1;
    }
    assert_eq!(count, 152, "decode configurations");
}

/// Every configuration in F32 and BF16 metadata, 152 cases per kernel: QuantizedGemv over 3 x 33 x 129 and QuantizedGemm
/// over 65 x 66 x 97 (m x n x k), crossing a 64-row tile, U4 codes crossing words and rows, partial groups and the
/// padded GroupOutput rows.
fn decode(gemm: bool) {
    let (m, n, k) = if gemm {
        (65, 66, 97)
    } else {
        (3, 33, 129)
    };
    let cases = itertools::iproduct!(configurations().into_iter().enumerate(), [DataType::F32, DataType::BF16]).map(
        |((index, configuration), metadata)| {
            let seed = index as u32 * 2 + u32::from(metadata == DataType::BF16);
            quantized(Case::new([metadata, DataType::F32, DataType::F32], m, n, k, 0, seed), configuration, seed.into())
        },
    );
    assert_eq!(check_all("decode", gemm, cases), 152, "decode cases");
}

#[uzu_test]
fn decode_gemv() {
    decode(false);
}

#[uzu_test]
fn decode_gemm() {
    decode(true);
}

/// Every triple under every flag mask, 32 for QuantizedGemv (256 cases, gathered outputs naming B rows among n + 3) over
/// 2 x 9 x 70 and 16 for QuantizedGemm (128 cases) over 65 x 9 x 40, the configuration rotating through all 76.
fn masks(gemm: bool) {
    let configurations = configurations();
    let (m, n, k, count) = if gemm {
        (65, 9, 40, 16)
    } else {
        (2, 9, 70, 32)
    };
    let cases = itertools::iproduct!(Case::triples().enumerate(), 0..count).map(|((triple, types), mask)| {
        let index = triple as u32 * count + mask;
        let configuration = configurations[index as usize % configurations.len()];
        quantized(Case::new(types, m, n, k, mask, index), configuration, index.into())
    });
    assert_eq!(check_all("masks", gemm, cases), 8 * count as usize, "mask cases");
}

#[uzu_test]
fn masks_gemv() {
    masks(false);
}

#[uzu_test]
fn masks_gemm() {
    masks(true);
}

/// K tails 1, 31, 33 and 257 under no and every flag; no rows or columns record nothing, leaving every guard and input
/// unchanged; K = 0 is the epilogue on +0, exactly +0 without flags and under every flag against the CPU and bounds; and two accumulating dispatches in one command buffer over exact integers
/// (unit scales, symmetric U4 codes) store D + 2 A Bᵀ bit for bit.
fn shapes_and_tails(gemm: bool) {
    let run = if gemm {
        Case::quantized_gemm
    } else {
        Case::quantized_gemv
    };
    let all = if gemm {
        15
    } else {
        31
    };
    let types = [[DataType::BF16; 3], [DataType::F32; 3], [DataType::BF16, DataType::F32, DataType::F32]];
    let configuration = (4, 32, ScaleZeroPoint, OutputGroup, true);
    let cases = itertools::iproduct!(types, [1, 31, 33, 257], [0, all])
        .map(|(types, k, mask)| quantized(Case::new(types, 65, 9, k, mask, k), configuration, k.into()));
    assert_eq!(check_all("tails", gemm, cases), 24, "tail cases");
    let fixture = KernelFixture::new();
    for case in [Case::new([DataType::BF16; 3], 0, 5, 7, all, 1), Case::new([DataType::F32; 3], 3, 0, 7, all, 2)] {
        assert!(run(&quantized(case, configuration, 1), &fixture, 1).is_empty(), "empty dispatch");
    }
    for (index, types) in Case::triples().enumerate() {
        let case = quantized(Case::new(types, 2, 3, 0, 0, 4), (8, 32, ScaleBias, GroupOutput, false), index as u64);
        case.exact(&fixture, "K 0", run, &[0.0; 6]);
    }
    let flagged = Case::triples().enumerate().map(|(index, types)| {
        quantized(
            Case::new(types, 2, 3, 0, all, 5 + index as u32),
            (4, 64, ScaleZeroPoint, OutputGroup, true),
            index as u64,
        )
    });
    assert_eq!(check_all("K 0 flags", gemm, flagged), 8, "flagged K 0 cases");
    let mut case = quantized(
        Case::new([DataType::BF16, DataType::BF16, DataType::F32], 70, 67, 40, Case::ACCUMULATE, 6),
        (4, 32, ScaleSymmetric, OutputGroup, false),
        6,
    );
    case.quantized.as_mut().unwrap().scales.fill(1.0);
    case.b = case.decoded();
    case.a = (0..case.a.len()).map(|index| (index % 5) as f32 - 2.0).collect();
    case.d = (0..case.d.len()).map(|index| index as f32).collect();
    let (n, k) = (case.n as usize, case.k as usize);
    let expected = (0..case.d.len())
        .map(|index| {
            let (row, column) = (index / n, index % n);
            let dot = case.a[row * k..][..k].iter().zip(&case.b[column * k..][..k]).map(|(x, w)| x * w).sum::<f32>();
            case.d[index] + 2.0 * dot
        })
        .collect::<Vec<_>>();
    KernelFixture::assert_bits(&expected, &case.cpu(2, false), "CPU chain");
    KernelFixture::assert_bits(&expected, &run(&case, &fixture, 2), "Vulkan chain");
    fixture.assert_clean();
}

#[uzu_test]
fn shapes_and_tails_gemv() {
    shapes_and_tails(false);
}

#[uzu_test]
fn shapes_and_tails_gemm() {
    shapes_and_tails(true);
}

/// Model layers 0.8b_qkv, 0.8b_down and 4b_down of qwen3_layer_shapes(8), at m 1 and 2 through QuantizedGemv and m 64
/// through QuantizedGemm, BF16 throughout, U4 group 64 biases in GroupOutput (MLX) and U8 group 128 zero points in
/// OutputGroup; and a tied readout gathering 8 outputs per row from the 151936-row vocabulary, 1024 columns.
fn models(gemm: bool) {
    let configurations = [(4, 64, ScaleBias, GroupOutput, false), (8, 128, ScaleZeroPoint, OutputGroup, false)];
    let layers = ["0.8b_qkv", "0.8b_down", "4b_down"];
    let shapes = qwen3_layer_shapes(8)
        .filter(|(label, shape)| {
            layers.contains(label)
                && if gemm {
                    shape.m == 64
                } else {
                    shape.m <= 2
                }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        shapes.len(),
        if gemm {
            3
        } else {
            6
        },
        "model shapes"
    );
    let mut cases = itertools::iproduct!(shapes, configurations)
        .map(|((_, shape), configuration)| {
            let case = Case::new([DataType::BF16; 3], shape.m, shape.n, shape.k, Case::SCALE | Case::BIAS, shape.m);
            quantized(case, configuration, shape.k.into())
        })
        .collect::<Vec<_>>();
    if !gemm {
        let (vocabulary, k) = (151_936u32, 1024);
        let mut readout = Case::new([DataType::BF16; 3], 2, 8, k, Case::GATHER, 7);
        readout.b = vec![0.0; (vocabulary * k) as usize];
        readout.gather = Some(
            (0..16)
                .map(|index| {
                    if index == 5 {
                        vocabulary - 1
                    } else {
                        (index * 40_503 + 11) % vocabulary
                    }
                })
                .collect(),
        );
        cases.push(quantized(readout, configurations[0], 7));
    }
    let expected = if gemm {
        6
    } else {
        13
    };
    assert_eq!(check_all("models", gemm, cases), expected, "model cases");
}

#[uzu_test]
fn models_gemv() {
    models(false);
}

#[uzu_test]
fn models_gemm() {
    models(true);
}

/// A 1 x n x k case whose weight row r holds the logical codes `codes[r]` (before any stored-sign flip) in one group of
/// scale `scales[r]` and correction `corrections[r]`: the bias, or for ScaleZeroPoint the zero point; `b` the decoded
/// weights.
fn witness(
    types: [DataType; 3],
    (bits, method, signed): (u32, QuantizationMethod, bool),
    a: &[f32],
    codes: &[&[u32]],
    scales: &[f32],
    corrections: &[f32],
    mask: u32,
) -> Case {
    let k = a.len();
    assert!(k <= 32, "one group per row");
    let rows = vec![vec![0.0; k]; codes.len()];
    let rows = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut case = Case::witness(types, a, &rows, mask).quantize(bits, 32, method, OutputGroup, signed, 0);
    let input = case.quantized.as_mut().unwrap();
    input.w_packed.fill(0);
    for (index, &code) in codes.iter().flat_map(|row| row.iter()).enumerate() {
        input.w_packed[index * bits as usize / 32] |= code << (index * bits as usize % 32);
    }
    for value in scales.iter().chain(corrections) {
        assert_eq!(Case::stored(*value, types[0]).to_bits(), value.to_bits(), "{value:e} is not stored exactly");
    }
    input.scales = scales.to_vec();
    match method {
        ScaleBias => input.biases = Some(corrections.to_vec()),
        ScaleZeroPoint => input.zero_points = Some(corrections.iter().map(|&point| point as u8).collect()),
        ScaleSymmetric => {},
    }
    case.b = case.decoded();
    case
}

/// Exact decodes, CPU and Vulkan bit for bit (NaN any NaN), each one productive K 1 weight against A 1 unless noted:
/// - no fusing: F32 scale 1 + 2^-23, U8 code 255, bias -(255 + 2^-15): the rounded product cancels to +0, a fused
///   multiply-add leaves -2^-23;
/// - separate correction: scale 1 + 2^-23, code 3, zero point 1: (3 + 2^-21) - (1 + 2^-23) rounds to 2 + 2^-21, bits
///   0x40000002, where the factored s (q - zp) is 2 + 2^-22;
/// - an FP32 weight from BF16 metadata: scale 1 + 2^-7, U8 code 255, symmetric: 256.9921875 - 129 = 127.9921875, bits
///   0x42fffc00, no BF16 value, stored by F32 D exactly and by BF16 D as its nearest 128;
/// - subnormal weights: F32 scale 2^-130 and BF16 2^-133, U4 code 3, symmetric: -5 2^-130 and -5 2^-133, against A
///   2^100 the normal -5 2^-30 and -5 2^-33, which a flushed weight loses;
/// - signed codes: U4 logical code 3, stored 11, symmetric unit scale: -5, where an unflipped code gives 3;
/// - nonfinite metadata: infinite scale times code 0 is NaN, an infinite scale's symmetric correction inf - inf NaN, a
///   finite scale 1.5 2^127 times code 255 overflows to +inf;
/// - signed zeros: scale -1, code 0, bias -0 decode -0, which the fold from +0 stores as +0, and scaled by -1 as -0.
fn exact_witnesses(gemm: bool) {
    let run = if gemm {
        Case::quantized_gemm
    } else {
        Case::quantized_gemv
    };
    let fixture = KernelFixture::new();
    let two = |exponent: i32| 2f32.powi(exponent);
    let (f32s, bf16s) = ([DataType::F32; 3], [DataType::BF16, DataType::F32, DataType::F32]);
    let fine = 1.0 + two(-23);
    let case = witness(f32s, (8, ScaleBias, false), &[1.0], &[&[255]], &[fine], &[-(255.0 + two(-15))], 0);
    assert_eq!(case.b[0].to_bits(), 0, "staged decode");
    case.exact(&fixture, "no fusing", run, &[0.0]);
    let case = witness(f32s, (8, ScaleZeroPoint, false), &[1.0], &[&[3]], &[fine], &[1.0], 0);
    assert_eq!(case.b[0].to_bits(), 0x4000_0002, "separate correction");
    case.exact(&fixture, "separate correction", run, &[f32::from_bits(0x4000_0002)]);
    for (d, expected) in [(DataType::F32, f32::from_bits(0x42ff_fc00)), (DataType::BF16, 128.0)] {
        let case = witness(
            [DataType::BF16, DataType::F32, d],
            (8, ScaleSymmetric, false),
            &[1.0],
            &[&[255]],
            &[1.0 + two(-7)],
            &[],
            0,
        );
        assert_eq!(case.b[0].to_bits(), 0x42ff_fc00, "decoded from BF16 metadata");
        case.exact(&fixture, "BF16 metadata", run, &[expected]);
    }
    for (types, scale, expected) in [(f32s, two(-130), -5.0 * two(-30)), (bf16s, two(-133), -5.0 * two(-33))] {
        let case = witness(types, (4, ScaleSymmetric, false), &[two(100)], &[&[3]], &[scale], &[], 0);
        assert_eq!(case.b[0], -5.0 * scale, "subnormal decode");
        case.exact(&fixture, "subnormal weight", run, &[expected]);
    }
    witness(f32s, (4, ScaleSymmetric, true), &[1.0], &[&[3]], &[1.0], &[], 0).exact(
        &fixture,
        "signed codes",
        run,
        &[-5.0],
    );
    let nonfinite =
        witness(f32s, (8, ScaleBias, false), &[1.0], &[&[0], &[255]], &[f32::INFINITY, 1.5 * two(127)], &[0.0, 0.0], 0);
    nonfinite.exact(&fixture, "nonfinite metadata", run, &[f32::NAN, f32::INFINITY]);
    witness(f32s, (8, ScaleSymmetric, false), &[1.0], &[&[9]], &[f32::INFINITY], &[], 0).exact(
        &fixture,
        "inf - inf",
        run,
        &[f32::NAN],
    );
    let mut zeros = witness(f32s, (8, ScaleBias, false), &[1.0], &[&[0]], &[-1.0], &[-0.0], 0);
    assert_eq!(zeros.b[0].to_bits(), 0x8000_0000, "decoded -0");
    zeros.exact(&fixture, "decoded -0", run, &[0.0]);
    zeros.ab_scale = Some(-1.0);
    zeros.exact(&fixture, "scaled -0", run, &[-0.0]);
    fixture.assert_clean();
}

#[uzu_test]
fn exact_witnesses_gemv() {
    exact_witnesses(false);
}

#[uzu_test]
fn exact_witnesses_gemm() {
    exact_witnesses(true);
}

/// Independent strides and one workgroup of every class. U4 zero points in OutputGroup over 3 groups (K 65, group 32) and
/// N 5: scales stride 3 per output, zero points 4 nibbles, the odd groups' in high nibbles. Then 8 outputs of 2 rows
/// over K 40 in one Gemv workgroup and one Gemm tile: weight row 1 of subnormal scales and zero biases, products the
/// careful path sums; row 2 of infinite scales, replayed; the others ordinary. Bounds or CPU classes, and CPU bits
/// through Gemm.
fn strides_and_classes(gemm: bool) {
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    for types in [[DataType::F32; 3], [DataType::BF16; 3]] {
        let case = quantized(Case::new(types, 2, 5, 65, 0, 9), (4, 32, ScaleZeroPoint, OutputGroup, false), 9);
        let (scales, zero_points) = case.strides();
        assert_eq!((scales.output_stride, zero_points.map(|strides| strides.output_stride)), (3, Some(4)), "strides");
        check(&fixture, "strides", &case, gemm, &mut totals);
        let mut mixed = quantized(Case::new(types, 2, 8, 40, 0, 5), (8, 32, ScaleBias, OutputGroup, false), 5);
        let input = mixed.quantized.as_mut().unwrap();
        input.scales[2..4].fill(Case::stored(2f32.powi(-130), types[0]));
        input.biases.as_mut().unwrap()[2..4].fill(0.0);
        input.scales[4..6].fill(f32::INFINITY);
        mixed.b = mixed.decoded();
        let (cpu, vulkan) = check(&fixture, "classes", &mixed, gemm, &mut totals);
        for (backend, values) in [("CPU", &cpu), ("Vulkan", &vulkan)] {
            for row in 0..2 {
                let [ordinary, tiny, replayed] = [0, 1, 2].map(|column| values[row * 8 + column]);
                assert!(ordinary.is_finite() && tiny != 0.0 && tiny.abs() < 1e-30, "{backend} {ordinary:e} {tiny:e}");
                assert!(!replayed.is_finite(), "{backend} replayed {replayed:e}");
            }
        }
    }
    Case::report(
        if gemm {
            "QuantizedGemm"
        } else {
            "QuantizedGemv"
        },
        &totals,
    );
    fixture.assert_clean();
}

#[uzu_test]
fn strides_and_classes_gemv() {
    strides_and_classes(false);
}

#[uzu_test]
fn strides_and_classes_gemm() {
    strides_and_classes(true);
}

/// Construction rejects group size 0, I8 codes and every data type but F32 and BF16.
#[uzu_test]
fn rejects_invalid_configurations() {
    let fixture = KernelFixture::new();
    let f32s = DataType::F32;
    for (mode, group, a) in
        [(QuantizationMode::U4, 0, f32s), (QuantizationMode::I8, 32, f32s), (QuantizationMode::U8, 32, DataType::F16)]
    {
        let gemv = QuantizedGemvVulkanKernel::new(
            &fixture.context,
            a,
            f32s,
            f32s,
            false,
            false,
            false,
            false,
            false,
            mode,
            ScaleBias,
            false,
            group,
        );
        let gemm = QuantizedGemmVulkanKernel::new(
            &fixture.context,
            a,
            f32s,
            f32s,
            false,
            false,
            false,
            false,
            mode,
            ScaleBias,
            false,
            group,
        );
        let expected = |result: &Result<_, Error>| match a {
            DataType::F16 => matches!(result, Err(Error::KernelVariant { .. })),
            _ => matches!(result, Err(Error::KernelPrecondition { .. })),
        };
        assert!(expected(&gemv.map(|_| ())) && expected(&gemm.map(|_| ())), "{mode:?} group {group} {a:?} accepted");
    }
    fixture.assert_clean();
}

/// Quantized against full-precision Matmul on the same A and D, dense B F32 holding the same decoded weights:
/// 0.8b_qkv (k 1024, n 3072) and 2b_up (k 2048, n 12288) at rows `ms`, BF16 metadata, A and D, U4 group 64 biases in
/// GroupOutput and U8 group 128 zero points in OutputGroup, hashed data and no flags; with `extras` an irregular 77 x
/// 3001 x 1001 tail and 0.8b_qkv at m 64 mixing careful outputs (A row 0 scaled by 2^-110) and replayed ones (a NaN
/// scale of row 5). Each case is first checked, quantized and dense, against the CPU and its bounds (and CPU bits
/// through Gemm), then timed in 4 interleaved quantized, dense pairs, each the median GPU and wall time of 10
/// submissions after 3 warm-up ones. Bytes count codes, metadata, A and D once: logical rates, not DRAM traffic.
/// With `a8`, the same layers with F32 A prepared as INT8 (A group 128, the canonical code layout of the bits), the
/// tail 77 x 3001 x 1056 with A group 32, and the mixed case's row 0 A scales times 2^-110, timed against the standard
/// quantized Matmul on the same decoded F32 A and packed B, whose outputs it must match (NaN any NaN); bytes count INT8
/// A and its FP32 scales.
fn throughput(
    gemm: bool,
    ms: &[u32],
    extras: bool,
    a8: bool,
) {
    let fixture = KernelFixture::new();
    let configurations = [(4, 64, ScaleBias, GroupOutput, false), (8, 128, ScaleZeroPoint, OutputGroup, false)];
    let types = if a8 {
        [DataType::BF16, DataType::F32, DataType::BF16]
    } else {
        [DataType::BF16; 3]
    };
    let prepare = |case: Case, group: u32| match a8 {
        true => {
            let bits = DataType::from(case.quantized.as_ref().unwrap().mode).size_in_bits() as u32;
            case.prepare_activations(group, Int8CodeLayout::for_right_bits(bits).unwrap(), None, false)
        },
        false => case,
    };
    let mut cases = Vec::new();
    for ((label, k, n), &m, configuration) in
        itertools::iproduct!([("0.8b_qkv", 1024, 3072), ("2b_up", 2048, 12288)], ms, configurations)
    {
        cases.push((label, prepare(quantized(Case::new(types, m, n, k, 0, m), configuration, m.into()), 128)));
    }
    if extras {
        let tail_k = if a8 {
            1056
        } else {
            1001
        };
        cases.push(("tail", prepare(quantized(Case::new(types, 77, 3001, tail_k, 0, 77), configurations[0], 77), 32)));
        let mut mixed = prepare(quantized(Case::new(types, 64, 3072, 1024, 0, 64), configurations[0], 64), 128);
        match mixed.quantized.as_mut().and_then(|input| input.prepared_a.as_mut()) {
            Some(prepared) => prepared.scales[..8].iter_mut().for_each(|scale| *scale *= 2f32.powi(-110)),
            None => mixed.a[..1024]
                .iter_mut()
                .for_each(|value| *value = Case::stored(*value * 2f32.powi(-110), DataType::BF16)),
        }
        if a8 {
            mixed.a = mixed.decoded_activations();
        }
        mixed.quantized.as_mut().unwrap().scales[5 * 16] = f32::NAN;
        mixed.b = mixed.decoded();
        cases.push(("0.8b_qkv mixed", mixed));
    }
    let mut totals = BTreeMap::new();
    let mut reference_totals = BTreeMap::new();
    let kernel = if gemm {
        "Gemm"
    } else {
        "Gemv"
    };
    let (name, reference_name, subject_label, reference_label, short) = match a8 {
        true => {
            (format!("A8Quantized{kernel}"), format!("Quantized{kernel}"), "A8", "quantized F32 A", ["A8", "quantized"])
        },
        false => (format!("Quantized{kernel}"), kernel.to_owned(), "quantized", "dense F32 B", ["quantized", "dense"]),
    };
    fn whole(buffer: &Arc<VkBuffer>) -> (&Arc<VkBuffer>, Range<u64>) {
        (buffer, 0..buffer.size())
    }
    for (label, case) in &cases {
        let mut reference = case.clone();
        match a8 {
            true => reference.quantized.as_mut().unwrap().prepared_a = None,
            false => (reference.quantized, reference.types[0]) = (None, DataType::F32),
        }
        let (_, subject_values) = check(&fixture, "throughput", case, gemm, &mut totals);
        let (_, reference_values) = match a8 {
            true => check(&fixture, "throughput", &reference, gemm, &mut reference_totals),
            false => {
                let run = if gemm {
                    Case::gemm
                } else {
                    Case::gemv
                };
                let (cpu, vulkan) = reference.check(&fixture, "throughput", run, &mut reference_totals);
                if gemm {
                    KernelFixture::assert_bits(&cpu, &vulkan, "dense CPU bits");
                }
                (cpu, vulkan)
            },
        };
        if a8 {
            KernelFixture::assert_bits(&reference_values, &subject_values, &format!("A8 {label} as standard"));
        }
        let planes = case.weight_planes().iter().map(|plane| fixture.buffer(plane)).collect::<Vec<_>>();
        let [a, d] = [(&reference.a, 1), (&case.d, 2)]
            .map(|(values, index)| fixture.buffer(&Case::bytes(values, reference.types[index])));
        let weights = planes.iter().map(whole).collect::<Vec<_>>();
        let (m, n, k) = (u64::from(case.m), u64::from(case.n), u64::from(case.k));
        let a_bytes =
            case.prepared().map_or(m * k * 2, |prepared| (prepared.values.len() + 4 * prepared.scales.len()) as u64);
        let bytes = planes.iter().map(|plane| plane.size()).sum::<u64>() + a_bytes + m * n * 2;
        let (reference, weights, a, d) = (&reference, &weights, &a, &d);
        // SAFETY (every recorder): whole buffers of the case's planes, INT8 or full-precision A, D and dense B; D aliases
        // nothing.
        let (mut subject, mut baseline): (
            Box<dyn FnMut(&mut VkCommandBufferEncoding) + '_>,
            Box<dyn FnMut(&mut VkCommandBufferEncoding) + '_>,
        ) = match (gemm, case.prepared()) {
            (false, None) => {
                let (subject, full) = (case.quantized_gemv_kernel(&fixture), reference.gemv_kernel(&fixture));
                let dense_b = fixture.buffer(&reference.b);
                (
                    Box::new(move |encoding| unsafe {
                        case.encode_quantized_gemv(&subject, weights, [whole(a), whole(d)], None, None, encoding)
                    }),
                    Box::new(move |encoding| unsafe {
                        reference.encode_gemv(&full, [whole(&dense_b), whole(a), whole(d)], None, None, encoding)
                    }),
                )
            },
            (true, None) => {
                let (subject, full) = (case.quantized_gemm_kernel(&fixture), reference.gemm_kernel(&fixture));
                let dense_b = fixture.buffer(&reference.b);
                (
                    Box::new(move |encoding| unsafe {
                        case.encode_quantized_gemm(&subject, weights, [whole(a), whole(d)], None, encoding)
                    }),
                    Box::new(move |encoding| unsafe {
                        reference.encode_gemm(&full, [whole(&dense_b), whole(a), whole(d)], None, encoding)
                    }),
                )
            },
            (false, Some(prepared)) => {
                let (subject, standard) =
                    (case.a8_quantized_gemv_kernel(&fixture), reference.quantized_gemv_kernel(&fixture));
                let (codes, scales) = (fixture.buffer(&prepared.values), fixture.buffer(&prepared.scales));
                (
                    Box::new(move |encoding| unsafe {
                        let int8 = [whole(&codes), whole(&scales), whole(d)];
                        case.encode_a8_gemv(&subject, weights, int8, None, None, encoding)
                    }),
                    Box::new(move |encoding| unsafe {
                        reference.encode_quantized_gemv(&standard, weights, [whole(a), whole(d)], None, None, encoding)
                    }),
                )
            },
            (true, Some(prepared)) => {
                let (subject, standard) =
                    (case.a8_quantized_gemm_kernel(&fixture), reference.quantized_gemm_kernel(&fixture));
                let (codes, scales) = (fixture.buffer(&prepared.values), fixture.buffer(&prepared.scales));
                (
                    Box::new(move |encoding| unsafe {
                        let int8 = [whole(&codes), whole(&scales), whole(d)];
                        case.encode_a8_gemm(&subject, weights, int8, None, encoding)
                    }),
                    Box::new(move |encoding| unsafe {
                        reference.encode_quantized_gemm(&standard, weights, [whole(a), whole(d)], None, encoding)
                    }),
                )
            },
        };
        // Even pairs time the subject first, odd pairs the reference; both are returned subject, reference.
        fn ordered<T>(
            pair: usize,
            mut subject: impl FnMut() -> T,
            mut reference: impl FnMut() -> T,
        ) -> [T; 2] {
            if pair.is_multiple_of(2) {
                let first = subject();
                [first, reference()]
            } else {
                let first = reference();
                [subject(), first]
            }
        }
        for pair in 0..4 {
            let times = ordered(pair, || fixture.median_times(&mut subject), || fixture.median_times(&mut baseline));
            let [[gpu, wall], [dense_gpu, dense_wall]] =
                times.map(|(gpu, wall)| [gpu, wall].map(|time| time.as_secs_f64() * 1e6));
            let first = short[pair % 2];
            eprintln!(
                "MEASURE {name} {label} {} m {m} n {n} k {k} pair {pair} ({first} first): {subject_label} GPU {gpu:.1} us \
                 wall {wall:.1} us, {reference_label} GPU {dense_gpu:.1} us wall {dense_wall:.1} us, {subject_label} \
                 {:.1} GB/s, {bytes} B",
                case.label(),
                bytes as f64 / gpu / 1e3,
            );
        }
    }
    Case::report(&name, &totals);
    Case::report(&reference_name, &reference_totals);
    fixture.assert_clean();
}

/// Run alone with `--ignored --nocapture`, as every quantized throughput test.
#[uzu_test]
#[ignore]
fn throughput_decode() {
    throughput(false, &[1, 2, 8], false, false);
}

#[uzu_test]
#[ignore]
fn throughput_m16() {
    throughput(true, &[16], false, false);
}

#[uzu_test]
#[ignore]
fn throughput_m64() {
    throughput(true, &[64], false, false);
}

#[uzu_test]
#[ignore]
fn throughput_m128() {
    throughput(true, &[128], true, false);
}

#[uzu_test]
#[ignore]
fn throughput_a8_decode() {
    throughput(false, &[1, 2, 8], false, true);
}

#[uzu_test]
#[ignore]
fn throughput_a8_m16() {
    throughput(true, &[16], false, true);
}

#[uzu_test]
#[ignore]
fn throughput_a8_m64() {
    throughput(true, &[64], false, true);
}

#[uzu_test]
#[ignore]
fn throughput_a8_m128() {
    throughput(true, &[128], true, true);
}
