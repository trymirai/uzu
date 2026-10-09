use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use uzu_engine_macros::uzu_test;

use super::{
    MatmulCase as Case, assert_quantized, check, check_all, kernel_fixture::KernelFixture, signs,
    transform_cpu_outputs, values,
};
use crate::{
    backends::{
        common::{
            Backend, Context, Kernels,
            gpu_types::{
                ActivationTransformOp,
                QuantizationMethod::{self, ScaleBias, ScaleSymmetric, ScaleZeroPoint},
                QuantizationMode::{self, I8, U4, U8},
            },
            kernel::matmul::{
                Int8CodeLayout::{self, GroupedByNibble, Sequential},
                MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulError, MatmulKernel, MatmulOutput, QuantParams,
                QuantParamsLayout::{self, GroupOutput, OutputGroup},
                QuantizedB, QuantizedCorrection,
            },
        },
        cpu::Cpu,
        vulkan::{
            A8QuantizedGemmVulkanKernel, A8QuantizedGemvVulkanKernel, ActivationTransformVulkanKernel, Error, VkBuffer,
        },
    },
    data_type::DataType::{self, BF16, F32},
    tests::{
        helpers::{create_buffer_with_data, create_context},
        matmul::qwen3_layer_shapes,
    },
};

const GROUPS: [u32; 3] = [32, 64, 128];

fn run(gemm: bool) -> fn(&Case, &KernelFixture, usize) -> Vec<f32> {
    if gemm {
        Case::a8_quantized_gemm
    } else {
        Case::a8_quantized_gemv
    }
}

/// The 36 logical configurations (bits, A code layout, A group, B group, method, metadata layout, stored signed B
/// codes, metadata type), bits outermost, then layout, A group and B group, configuration i taking method i % 3,
/// metadata layout (i / 3) % 2, signed (i / 6) % 2 and BF16 metadata (i / 12) % 2. Asserts the rotation's actual
/// coverage: in each bit's 18 configurations methods 6/6/6, metadata layouts 9/9, unsigned/signed 12/6 for U4 and 6/12
/// for U8, F32/BF16 metadata 12/6 and 18 distinct method/layout/sign/metadata combinations; indices 0 to 23 all 24.
fn configurations() -> Vec<(u32, Int8CodeLayout, u32, u32, QuantizationMethod, QuantParamsLayout, bool, DataType)> {
    let configurations = itertools::iproduct!([4, 8], [Sequential, GroupedByNibble], GROUPS, GROUPS)
        .enumerate()
        .map(|(i, (bits, layout, a_group, b_group))| {
            let method = [ScaleBias, ScaleZeroPoint, ScaleSymmetric][i % 3];
            (
                bits,
                layout,
                a_group,
                b_group,
                method,
                [OutputGroup, GroupOutput][i / 3 % 2],
                i / 6 % 2 == 1,
                [F32, BF16][i / 12 % 2],
            )
        })
        .collect::<Vec<_>>();
    let combination = |c: &(u32, Int8CodeLayout, u32, u32, QuantizationMethod, QuantParamsLayout, bool, DataType)| {
        format!("{:?} {:?} {} {:?}", c.4, c.5, c.6, c.7)
    };
    for (bits, signed) in [(4, [12, 6]), (8, [6, 12])] {
        let half = configurations.iter().filter(|c| c.0 == bits);
        assert_eq!(
            [ScaleBias, ScaleZeroPoint, ScaleSymmetric].map(|m| half.clone().filter(|c| c.4 == m).count()),
            [6; 3]
        );
        assert_eq!([OutputGroup, GroupOutput].map(|l| half.clone().filter(|c| c.5 == l).count()), [9, 9]);
        assert_eq!([false, true].map(|s| half.clone().filter(|c| c.6 == s).count()), signed, "U{bits} signed");
        assert_eq!([F32, BF16].map(|t| half.clone().filter(|c| c.7 == t).count()), [12, 6], "U{bits} metadata");
        assert_eq!(half.map(combination).collect::<BTreeSet<_>>().len(), 18, "U{bits} combinations");
    }
    assert_eq!(configurations[..24].iter().map(combination).collect::<BTreeSet<_>>().len(), 24, "indices 0..24");
    configurations
}

/// The case quantized as the configuration's B and its A prepared by the canonical CPU producer, without sums.
fn a8(
    case: Case,
    (bits, layout, a_group, b_group, method, params, signed, _): (
        u32,
        Int8CodeLayout,
        u32,
        u32,
        QuantizationMethod,
        QuantParamsLayout,
        bool,
        DataType,
    ),
    seed: u64,
) -> Case {
    case.quantize(bits, b_group, method, params, signed, seed).prepare_activations(a_group, layout, None, signed)
}

/// Every configuration through A8QuantizedGemv at m 1 and 2 and A8QuantizedGemm at m 65, n 33 and K three A groups,
/// so the A groups 32 and 64 under larger B groups end in partial B groups: 108 dispatches.
fn decode(gemm: bool) {
    let ms: &[u32] = if gemm {
        &[65]
    } else {
        &[1, 2]
    };
    let cases = itertools::iproduct!(configurations().into_iter().enumerate(), ms).map(|((index, c), &m)| {
        let seed = index as u32 * 3 + m;
        a8(Case::new([c.7, F32, F32], m, 33, 3 * c.2, 0, seed), c, seed.into())
    });
    assert_eq!(check_all("A8 decode", gemm, cases), 36 * ms.len(), "decode cases");
}

#[uzu_test]
fn decode_gemv() {
    decode(false);
}

#[uzu_test]
fn decode_gemm() {
    decode(true);
}

/// The CPU's INT8 A is its full-precision A of the decoded values: for every configuration, the CPU on the prepared
/// codes and scales stores the bits of the CPU on `a` (NaN any NaN) with the same packed B. Then, finite only, an
/// identity B (U8 symmetric codes 129 and 128, unit scales) stores each decoded activation.
#[uzu_test]
fn cpu_decodes_prepared_activations() {
    for (index, c) in configurations().into_iter().enumerate() {
        let case = a8(Case::new([c.7, F32, F32], 3, 33, 3 * c.2, 0, index as u32), c, index as u64);
        let mut full = case.clone();
        full.quantized.as_mut().unwrap().prepared_a = None;
        KernelFixture::assert_bits(&full.cpu(1, false), &case.cpu(1, false), &case.label());
    }
    let k = 96;
    for layout in [Sequential, GroupedByNibble] {
        let mut case =
            a8(Case::new([F32; 3], 3, k, k, 0, 1), (8, layout, 32, 32, ScaleSymmetric, OutputGroup, false, F32), 1);
        let input = case.quantized.as_mut().unwrap();
        let codes = (0..k * k)
            .map(|index| {
                if index / k == index % k {
                    129u8
                } else {
                    128
                }
            })
            .collect::<Vec<_>>();
        input.w_packed = bytemuck::pod_collect_to_vec(&codes);
        input.scales.fill(1.0);
        case.b = case.decoded();
        for (index, (&stored, &decoded)) in case.cpu(1, false).iter().zip(&case.a).enumerate() {
            assert!(decoded.is_finite(), "decoded {decoded:e}");
            assert!(stored.to_bits() == decoded.to_bits() || (stored == 0.0 && decoded == 0.0), "{index}: {stored:e}");
        }
    }
}

/// Every B/D pair under every flag mask, 32 for A8QuantizedGemv (gathered outputs naming B rows among n + 3) over m 2
/// and 16 for A8QuantizedGemm over m 65, n 9, the configuration rotating through all 36.
fn masks(gemm: bool) {
    let configurations = configurations();
    let (m, count) = if gemm {
        (65, 16)
    } else {
        (2, 32)
    };
    let cases = itertools::iproduct!(itertools::iproduct!([F32, BF16], [F32, BF16]).enumerate(), 0..count).map(
        |((pair, (b, d)), mask)| {
            let index = pair as u32 * count + mask;
            let c = configurations[index as usize % configurations.len()];
            a8(Case::new([b, F32, d], m, 9, 3 * c.2, mask, index), c, index.into())
        },
    );
    assert_eq!(check_all("A8 masks", gemm, cases), 4 * count as usize, "mask cases");
}

#[uzu_test]
fn masks_gemv() {
    masks(false);
}

#[uzu_test]
fn masks_gemm() {
    masks(true);
}

/// K 32, 96, 160 and 1056 with A group 32 and B groups 64 and 128 (partial B groups), M and N tails, without and under
/// every flag. No rows (empty A, the producer's descriptor only) or no columns (real preparation) record nothing,
/// leaving every guard and input unchanged. K 0 folds from +0: exactly +0 without flags, and under every flag the
/// canonical epilogue against the CPU and bounds. Two accumulating dispatches in one command buffer over exact integers
/// (unit scales, codes -2 to 2, symmetric U4) store D + 2 A Bᵀ bit for bit.
fn shapes_and_tails(gemm: bool) {
    let run = run(gemm);
    let (m, all) = if gemm {
        (65, 15)
    } else {
        (3, 31)
    };
    let cases = itertools::iproduct!([32, 96, 160, 1056], [64, 128], [0, all]).map(|(k, b_group, mask)| {
        let c = (4, GroupedByNibble, 32, b_group, ScaleZeroPoint, OutputGroup, true, BF16);
        a8(Case::new([BF16, F32, BF16], m, 33, k, mask, k + b_group), c, k.into())
    });
    assert_eq!(check_all("A8 tails", gemm, cases), 16, "tail cases");
    let fixture = KernelFixture::new();
    let c = (8, Sequential, 64, 32, ScaleBias, GroupOutput, false, F32);
    for (m, n) in [(0, 5), (3, 0)] {
        assert!(run(&a8(Case::new([BF16, F32, BF16], m, n, 128, all, 1), c, 1), &fixture, 1).is_empty(), "empty");
    }
    let pairs = itertools::iproduct!([F32, BF16], [F32, BF16]).enumerate().collect::<Vec<_>>();
    for &(index, (b, d)) in &pairs {
        a8(Case::new([b, F32, d], 2, 3, 0, 0, 4), c, index as u64).exact(&fixture, "K 0", run, &[0.0; 6]);
    }
    let flagged = pairs.iter().map(|&(index, (b, d))| a8(Case::new([b, F32, d], 2, 3, 0, all, 5 + index as u32), c, 5));
    assert_eq!(check_all("A8 K 0 flags", gemm, flagged), 4, "flagged K 0 cases");
    let mut case = a8(
        Case::new([BF16, F32, F32], 70, 67, 64, Case::ACCUMULATE, 6),
        (4, Sequential, 32, 32, ScaleSymmetric, OutputGroup, false, BF16),
        6,
    );
    let input = case.quantized.as_mut().unwrap();
    input.scales.fill(1.0);
    let prepared = input.prepared_a.as_mut().unwrap();
    prepared.values.iter_mut().enumerate().for_each(|(index, code)| *code = (index % 5) as i8 - 2);
    prepared.scales.fill(1.0);
    (case.a, case.b) = (case.decoded_activations(), case.decoded());
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

/// Model layers 0.8b_qkv, 0.8b_down and 4b_down of qwen3_layer_shapes(8), at m 1 and 2 through A8QuantizedGemv and m
/// 64 through A8QuantizedGemm, BF16 B and D, scale and bias, U4 group 64 biases in GroupOutput, A group 128 grouped by
/// weight nibble.
fn models(gemm: bool) {
    let layers = ["0.8b_qkv", "0.8b_down", "4b_down"];
    let cases = qwen3_layer_shapes(8)
        .filter(|(label, shape)| {
            layers.contains(label)
                && if gemm {
                    shape.m == 64
                } else {
                    shape.m <= 2
                }
        })
        .map(|(_, shape)| {
            let case = Case::new([BF16, F32, BF16], shape.m, shape.n, shape.k, Case::SCALE | Case::BIAS, shape.m);
            a8(case, (4, GroupedByNibble, 128, 64, ScaleBias, GroupOutput, false, BF16), shape.k.into())
        });
    let expected = if gemm {
        3
    } else {
        6
    };
    assert_eq!(check_all("A8 models", gemm, cases), expected, "model cases");
}

#[uzu_test]
fn models_gemv() {
    models(false);
}

#[uzu_test]
fn models_gemm() {
    models(true);
}

/// F32 throughout: A rows of logical INT8 `codes` with one scale per A group in `scales`, stored in `layout`, against
/// one B row of U8 symmetric `b_codes` (weight code - 128) of scale `b_scale` in groups of `b_group`; `a` and `b` the
/// decoded values.
fn witness(
    codes: &[Vec<i8>],
    scales: &[Vec<f32>],
    (a_group, layout): (u32, Int8CodeLayout),
    b_codes: &[u8],
    (b_scale, b_group): (f32, u32),
) -> Case {
    let (m, k) = (codes.len(), b_codes.len());
    let case =
        Case::new([F32; 3], m as u32, 1, k as u32, 0, 0).quantize(8, b_group, ScaleSymmetric, OutputGroup, false, 0);
    let mut case = case.prepare_activations(a_group, layout, None, false);
    let input = case.quantized.as_mut().unwrap();
    input.w_packed = bytemuck::pod_collect_to_vec(b_codes);
    input.scales.fill(b_scale);
    let prepared = input.prepared_a.as_mut().unwrap();
    for (row, logical) in codes.iter().enumerate() {
        for (inner, &code) in logical.iter().enumerate() {
            prepared.values[row * k + layout.index(inner)] = code;
        }
    }
    prepared.scales = scales.concat();
    (case.a, case.b) = (case.decoded_activations(), case.decoded());
    case
}

/// Exact decodes, CPU and Vulkan bit for bit (NaN any NaN), against B weights 1 (code 129) unless noted:
/// - signed extremes: codes -128 and 127 sum to -1, where unsigned codes give 255;
/// - the permutation: logical codes j + 1 against weights j over K 32 store Σ (j + 1) j in both layouts;
/// - the A group: K 128, A group 32, scales 1, 2, 4 and 8, codes 1: 480 under B groups 32, 64 and 128 (a scale read at
///   the B group would give 192 for B 64 and 128 for B 128);
/// - a subnormal A scale: 2^-130 times code 3 against a weight 2^100 is the normal 3 2^-30;
/// - classes: an infinite scale times code 0 is NaN, times codes 1 and -1 sums to ±inf, a NaN scale NaN;
/// - signed zeros: a host decode of code 0 and scale -1 is -0; a +0 dot scaled by -1 stores -0, onto a prior -0 still -0,
///   onto a prior +0 +0.
fn witnesses(gemm: bool) {
    let run = run(gemm);
    let fixture = KernelFixture::new();
    let ones = |k: usize| vec![129u8; k];
    let row = |k: usize, lead: &[i8]| [lead, &vec![0; k - lead.len()][..]].concat();
    witness(&[row(32, &[-128, 127])], &[vec![1.0]], (32, Sequential), &ones(32), (1.0, 32)).exact(
        &fixture,
        "extremes",
        run,
        &[-1.0],
    );
    let weights = (0..32).map(|j| 128 + j as u8).collect::<Vec<_>>();
    let codes = (0..32).map(|j| j as i8 + 1).collect::<Vec<_>>();
    let expected = (0..32).map(|j| ((j + 1) * j) as f32).sum::<f32>();
    for layout in [Sequential, GroupedByNibble] {
        witness(std::slice::from_ref(&codes), &[vec![1.0]], (32, layout), &weights, (1.0, 32)).exact(
            &fixture,
            "permutation",
            run,
            &[expected],
        );
    }
    for b_group in GROUPS {
        witness(&[vec![1; 128]], &[vec![1.0, 2.0, 4.0, 8.0]], (32, Sequential), &ones(128), (1.0, b_group)).exact(
            &fixture,
            &format!("A group under B group {b_group}"),
            run,
            &[480.0],
        );
    }
    let two = |exponent: i32| 2f32.powi(exponent);
    let tiny = witness(&[row(32, &[3])], &[vec![two(-130)]], (32, Sequential), &ones(32), (two(100), 32));
    assert_eq!((tiny.a[0], tiny.b[0]), (3.0 * two(-130), two(100)), "subnormal decode");
    tiny.exact(&fixture, "subnormal A scale", run, &[3.0 * two(-30)]);
    let scales = [f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::NAN].map(|scale| vec![scale]);
    let codes = [0, 1, -1, 1].map(|code| vec![code; 32]);
    witness(&codes, &scales, (32, Sequential), &ones(32), (1.0, 32)).exact(
        &fixture,
        "classes",
        run,
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    );
    let negative = witness(&[vec![0; 32]], &[vec![-1.0]], (32, Sequential), &ones(32), (1.0, 32));
    assert_eq!(negative.a[0].to_bits(), 0x8000_0000, "decoded -0");
    let mut zeros = witness(&[vec![0; 32]], &[vec![1.0]], (32, Sequential), &ones(32), (1.0, 32));
    zeros.ab_scale = Some(-1.0);
    zeros.exact(&fixture, "scaled -0", run, &[-0.0]);
    zeros.accumulate = true;
    for prior in [-0.0, 0.0] {
        zeros.d = vec![prior];
        zeros.exact(&fixture, &format!("-0 onto {prior:e}"), run, &[prior]);
    }
    fixture.assert_clean();
}

#[uzu_test]
fn witnesses_gemv() {
    witnesses(false);
}

#[uzu_test]
fn witnesses_gemm() {
    witnesses(true);
}

/// Group sums are not part of A8's product: the CPU stores the same bits with the producer's sums and with garbage
/// ones; the Vulkan entries take no sums argument at all.
#[uzu_test]
fn group_sums_are_ignored() {
    let case = Case::new([F32; 3], 3, 33, 128, 0, 1).quantize(4, 32, ScaleBias, OutputGroup, false, 1);
    let case = case.prepare_activations(64, GroupedByNibble, Some(32), false);
    assert_eq!(case.prepared().unwrap().group_sums.len(), 12, "sums of 32 codes");
    let mut garbage = case.clone();
    garbage.quantized.as_mut().unwrap().prepared_a.as_mut().unwrap().group_sums.fill(i32::MIN);
    KernelFixture::assert_bits(&case.cpu(1, false), &garbage.cpu(1, false), "garbage sums");
}

/// Whether the CPU MatmulKernel rejects INT8 A of `a_group` codes over K `k` against `b` (U4/U8/I8 codes of a group,
/// else full precision), before reading anything; any error but its canonical `MatmulError::IncompatibleA` fails.
fn cpu_rejects(
    a_group: u32,
    k: u32,
    b: Option<(QuantizationMode, u32)>,
) -> bool {
    let context = create_context::<Cpu>();
    let mut kernel =
        <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(&context, F32, F32, F32).expect("CPU MatmulKernel");
    let buffer = || create_buffer_with_data::<Cpu, u8>(&context, &[0; 4096]);
    let (values, scales, codes, b_scales, mut d) = (buffer(), buffer(), buffer(), buffer(), buffer());
    let arguments = MatmulArguments {
        a: MatmulA::Int8Symmetric {
            values: &values,
            scales: &scales,
            group_sums: None,
            scale_group_size: a_group,
            code_layout: Sequential,
        },
        b: match b {
            None => MatmulB::FullPrecision {
                b: &codes,
            },
            Some((mode, group)) => MatmulB::Quantized(QuantizedB {
                codes: &codes,
                scales: &b_scales,
                correction: QuantizedCorrection::Symmetric,
                params: QuantParams::new(OutputGroup, 1, k.div_ceil(group)),
                mode,
                group_size: group,
                signed_codes: false,
            }),
        },
        b_leading_dimension: None,
        b_transpose: true,
        output: MatmulOutput::new(&mut d, MatmulDOps::none()),
        gather_indices: None::<&<Cpu as Backend>::GlobalBuffer>,
        m: 1,
        n: 1,
        k,
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    let Err(error) = kernel.encode(arguments, &mut command_buffer) else {
        return false;
    };
    let matmul = std::error::Error::source(&error).and_then(|source| source.downcast_ref::<MatmulError<Cpu>>());
    assert!(
        matches!(
            matmul,
            Some(MatmulError::IncompatibleA {
                path: "CpuMatmul",
                ..
            })
        ),
        "unexpected CPU error {error:?}"
    );
    true
}

/// The CPU rejects A groups 0, 16, 48 and 256, B groups 16, 48 and 256, K not a multiple of the A group, I8 B codes and
/// full-precision B, and admits the valid control. Vulkan construction rejects the same groups and I8 codes; `encode`
/// rejects K not a multiple of the A group before recording, and the command buffer then completes with nothing written.
#[uzu_test]
fn rejects_invalid_contracts() {
    for a_group in [0, 16, 48, 256] {
        assert!(cpu_rejects(a_group, 3 * a_group.max(32), Some((U4, 32))), "CPU A group {a_group}");
    }
    for b_group in [16, 48, 256] {
        assert!(cpu_rejects(32, 96, Some((U8, b_group))), "CPU B group {b_group}");
    }
    assert!(cpu_rejects(32, 104, Some((U4, 32))), "CPU K 104");
    assert!(cpu_rejects(32, 96, Some((I8, 32))), "CPU I8");
    assert!(cpu_rejects(32, 96, None), "CPU full-precision B");
    assert!(!cpu_rejects(32, 96, Some((U4, 32))), "CPU control");
    let fixture = KernelFixture::new();
    let new = |mode, b_group, a_group| {
        let gemv = A8QuantizedGemvVulkanKernel::new(
            &fixture.context,
            F32,
            F32,
            false,
            false,
            false,
            false,
            false,
            mode,
            ScaleSymmetric,
            false,
            b_group,
            a_group,
            false,
        );
        let gemm = A8QuantizedGemmVulkanKernel::new(
            &fixture.context,
            F32,
            F32,
            false,
            false,
            false,
            false,
            mode,
            ScaleSymmetric,
            false,
            b_group,
            a_group,
            false,
        );
        (gemv, gemm)
    };
    let invalid = [(U4, 32, 0), (U4, 32, 16), (U4, 32, 48), (U4, 32, 256), (U8, 16, 32), (U8, 48, 32), (U8, 256, 32)];
    for (mode, b_group, a_group) in invalid.into_iter().chain([(I8, 32, 32)]) {
        let (gemv, gemm) = new(mode, b_group, a_group);
        let rejected = |result: Result<(), Error>| matches!(result, Err(Error::KernelPrecondition { .. }));
        assert!(rejected(gemv.map(|_| ())) && rejected(gemm.map(|_| ())), "{mode:?} groups {b_group}/{a_group}");
    }
    let (gemv, gemm) = new(U4, 32, 32);
    let (gemv, gemm) = (gemv.expect("A8QuantizedGemv"), gemm.expect("A8QuantizedGemm"));
    let untouched = fixture.buffer(&[0u8; 4096]);
    let whole = || (&untouched, 0..4096);
    let mut encoding = fixture.encoding();
    // SAFETY: never dispatched: the precondition fails before recording.
    let gemv_rejected = catch_unwind(AssertUnwindSafe(|| unsafe {
        gemv.encode(
            whole(),
            whole(),
            None,
            None,
            whole(),
            whole(),
            whole(),
            None,
            None,
            104,
            1,
            1,
            1,
            1,
            None,
            None,
            None,
            None,
            &mut encoding,
        )
    }));
    // SAFETY: as above.
    let gemm_rejected = catch_unwind(AssertUnwindSafe(|| unsafe {
        gemm.encode(
            whole(),
            whole(),
            None,
            None,
            whole(),
            whole(),
            whole(),
            None,
            104,
            1,
            1,
            1,
            1,
            None,
            None,
            None,
            None,
            &mut encoding,
        )
    }));
    assert!(gemv_rejected.is_err() && gemm_rejected.is_err(), "K 104 accepted");
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded nothing.
    assert!(unsafe { KernelFixture::read::<u8>(&untouched) }.iter().all(|&byte| byte == 0), "a rejected call wrote");
    fixture.assert_clean();
}

/// The real producer and A8 in one command buffer, twice: ActivationTransform quantizes guarded F32 rows with ±1
/// Hadamard factors into sentinel-guarded code, scale and (every other case) group-sum ranges allocated outside the
/// harness, and A8 reads those ranges through the recorded access barriers, the second producer write following the
/// first A8 read. The harness uploads sentinel A planes, which A8 never reads. Asserted in order: the producer's outputs
/// are the CPU producer's exactly, every guard, input and factor unchanged; then D is bit for bit A8 on the CPU
/// producer's planes, which `check` holds to the CPU producer and MatmulKernel and their bounds (and CPU bits through
/// Gemm). K 96 with A group 32 and K 384 with every admitted A >= B group pair, both code layouts, A8QuantizedGemv at m 1
/// and 3 and A8QuantizedGemm at m 65.
fn producer(gemm: bool) {
    const CODE: i8 = 0x5a;
    const SCALE: f32 = -7.0;
    const SUM: i32 = 0x5a5a_5a5a;
    fn range((buffer, range): &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
        (buffer, range.clone())
    }
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    let ms: &[u32] = if gemm {
        &[65]
    } else {
        &[1, 3]
    };
    let groups =
        [(96, 32, 32), (384, 32, 32), (384, 64, 32), (384, 64, 64), (384, 128, 32), (384, 128, 64), (384, 128, 128)];
    let mut count = 0;
    for (index, (layout, (k, a_group, b_group), &m)) in
        itertools::iproduct!([Sequential, GroupedByNibble], groups, ms).enumerate()
    {
        let (sums, signed) = (index % 2 == 0, index % 4 < 2);
        let bits = if layout == GroupedByNibble {
            4
        } else {
            8
        };
        let (x, factors) = (values::<f32>((m * k) as usize, index), signs(k as usize, index));
        let quantization = Some((a_group as usize, sums.then_some(b_group as usize), layout));
        let (_, codes, scales, group_sums) = transform_cpu_outputs((
            ActivationTransformOp::InputRht,
            false,
            None::<&[f32]>,
            quantization,
            &x[..],
            &factors[..],
            m as usize,
        ));
        let method = [ScaleBias, ScaleZeroPoint, ScaleSymmetric][index % 3];
        let case = Case::new([BF16, F32, F32], m, 33, k, Case::SCALE, index as u32);
        let mut case = case.quantize(bits, b_group, method, OutputGroup, signed, index as u64).prepare_activations(
            a_group,
            layout,
            sums.then_some(b_group),
            signed,
        );
        let prepared = case.quantized.as_mut().unwrap().prepared_a.as_mut().unwrap();
        (prepared.values, prepared.scales, prepared.group_sums) = (codes.clone(), scales.clone(), group_sums.clone());
        case.a = case.decoded_activations();
        let (_, expected) = check(&fixture, "A8 producer", &case, gemm, &mut totals);
        let mut harness = case.clone();
        let uploaded = harness.quantized.as_mut().unwrap().prepared_a.as_mut().unwrap();
        uploaded.values.fill(CODE);
        uploaded.scales.fill(SCALE);
        let elements = (m * k) as usize;
        let input = fixture.guarded(&x, SCALE);
        let factor_range = fixture.guarded(&factors, SUM);
        let codes_out = fixture.guarded(&vec![CODE; elements], CODE);
        let scales_out = fixture.guarded(&vec![SCALE; elements / a_group as usize], SCALE);
        let sums_out = sums.then(|| fixture.guarded(&vec![SUM; elements / b_group as usize], SUM));
        let ops = if sums {
            ActivationTransformOp::QuantizeWithGroupSums
        } else {
            ActivationTransformOp::Quantize
        };
        let sum_group = if sums {
            b_group
        } else {
            32
        };
        let grouped = layout.is_grouped_by_nibble();
        let transform = ActivationTransformVulkanKernel::new(
            &fixture.context,
            F32,
            F32,
            ops,
            grouped,
            false,
            a_group,
            sum_group,
            false,
        )
        .expect("Vulkan ActivationTransform");
        let gemv_kernel = (!gemm).then(|| case.a8_quantized_gemv_kernel(&fixture));
        let gemm_kernel = gemm.then(|| case.a8_quantized_gemm_kernel(&fixture));
        // SAFETY: guarded ranges of exactly the producer's rows, factors and outputs, which A8 reads after the
        // producer's write; the harness's B planes, D, bias and gather; D aliases nothing.
        let vulkan = harness.gpu(&fixture, 2, |weights, _, d, bias, gather, encoding| unsafe {
            transform.encode(
                Some(range(&input)),
                None,
                None,
                Some(range(&codes_out)),
                Some(range(&scales_out)),
                sums_out.as_ref().map(range),
                range(&factor_range),
                m,
                k,
                encoding,
            );
            let a = [range(&codes_out), range(&scales_out), d];
            match (&gemv_kernel, &gemm_kernel) {
                (Some(kernel), _) => case.encode_a8_gemv(kernel, weights, a, bias, gather, encoding),
                (_, Some(kernel)) => case.encode_a8_gemm(kernel, weights, a, bias, encoding),
                (None, None) => unreachable!("one kernel"),
            }
        });
        let label = format!("producer {}", case.label());
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe {
            KernelFixture::assert_unchanged(&input, SCALE, &x, "x");
            KernelFixture::assert_unchanged(&factor_range, SUM, &factors, "factors");
            let produced = (
                KernelFixture::read_guarded(&codes_out, CODE),
                KernelFixture::read_guarded(&scales_out, SCALE),
                sums_out.as_ref().map_or_else(Vec::new, |sums| KernelFixture::read_guarded(sums, SUM)),
            );
            assert_quantized(&produced, &(codes, scales, group_sums), &x, &label);
        }
        KernelFixture::assert_bits(&expected, &vulkan, &label);
        count += 1;
    }
    assert_eq!(count, 14 * ms.len(), "producer cases");
    Case::report(
        if gemm {
            "A8QuantizedGemm"
        } else {
            "A8QuantizedGemv"
        },
        &totals,
    );
    fixture.assert_clean();
}

#[uzu_test]
fn producer_gemv() {
    producer(false);
}

#[uzu_test]
fn producer_gemm() {
    producer(true);
}
