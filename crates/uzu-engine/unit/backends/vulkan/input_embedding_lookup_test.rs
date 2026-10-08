use std::{
    fmt::Debug,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Instant,
};

use bytemuck::NoUninit;
use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{InputEmbeddingLookupCase as Case, check_bounds, kernel_fixture::KernelFixture, round32, to};
use crate::{
    array::ArrayElement,
    backends::{
        common::gpu_types::{
            EmbeddingTableKind::{D4S4, Dense, Quantized},
            QuantizationMethod::{self, ScaleBias, ScaleSymmetric, ScaleZeroPoint},
            QuantizationMode::{self, I8, U4, U8},
        },
        vulkan::{Error, InputEmbeddingLookupVulkanKernel},
    },
    data_type::DataType,
};

const VOCAB: u32 = 5;
/// Duplicate rows, both ends of the vocabulary and both kinds of invalid token.
const TOKENS: [u32; 7] = [3, 0, VOCAB, 3, u32::MAX, VOCAB - 1, 1];
const MODES: [QuantizationMode; 3] = [U4, I8, U8];
const METHODS: [QuantizationMethod; 3] = [ScaleBias, ScaleZeroPoint, ScaleSymmetric];

/// Runs every case on Vulkan in one command buffer and on the CPU, then checks both against the staged oracle. Its
/// stages are single roundings of exact values, so the bounds are points: every element must match exactly, NaN by
/// class, zeros by sign.
fn check<T: ArrayElement + Float + NoUninit + Debug>(
    fixture: &KernelFixture,
    cases: &[Case<T>],
) {
    let outputs = Case::gpu(fixture, cases, fixture.encoding());
    assert_eq!(outputs.len(), cases.len(), "one output per case");
    for (case, gpu) in cases.iter().zip(outputs) {
        check_bounds(&case.oracle(), &case.cpu(), &gpu, &case.label());
    }
}

/// Dense rows of odd and tail lengths without the transform, raw transformed rows, an empty vocabulary whose rows are
/// all invalid, and empty batches and rows, which record nothing.
fn dense_matches_oracle<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let mut cases =
        [1, 7, 31, 129, 300].map(|dim| Case::<T>::new(Dense, None, (VOCAB, dim), &TOKENS, dim as usize)).to_vec();
    cases.extend([32, 128, 256].map(|dim| Case::new(Dense, None, (VOCAB, dim), &TOKENS, 1).hadamard(dim as usize)));
    cases.push(Case::new(Dense, None, (0, 64), &TOKENS, 2));
    cases.push(Case::new(Dense, None, (VOCAB, 64), &[], 2));
    cases.push(Case::new(Dense, None, (VOCAB, 0), &TOKENS, 3));
    check(&fixture, &cases);
    fixture.assert_clean();
}

#[uzu_test]
fn dense_matches_oracle_all_types() {
    dense_matches_oracle::<f32>();
    dense_matches_oracle::<bf16>();
}

/// Every mode and method over whole and partial groups, odd group counts (packed U4 zero points end mid-byte), odd
/// rows without U4, and transformed rows; an empty vocabulary, batch and row.
fn quantized_matches_oracle<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let mut cases = Vec::new();
    for (mode, method) in MODES.into_iter().flat_map(|mode| METHODS.map(|method| (mode, method))) {
        let quantized =
            |dim, group, seed| Case::<T>::new(Quantized, Some((mode, method, group)), (VOCAB, dim), &TOKENS, seed);
        cases.extend([(96, 32), (112, 48), (130, 7), (64, 1)].map(|(dim, group)| quantized(dim, group, dim as usize)));
        cases.extend(
            [(64, 64), (256, 128), (96, 48)].map(|(dim, group)| quantized(dim, group, 5).hadamard(dim as usize)),
        );
        if mode != U4 {
            cases.push(quantized(31, 4, 9));
        }
    }
    let quantization = Some((U4, ScaleZeroPoint, 32));
    cases.push(Case::new(Quantized, quantization, (0, 64), &TOKENS, 1).hadamard(1));
    cases.push(Case::new(Quantized, quantization, (VOCAB, 64), &[], 1).hadamard(1));
    cases.push(Case::new(Quantized, quantization, (VOCAB, 0), &TOKENS, 1));
    check(&fixture, &cases);
    fixture.assert_clean();
}

#[uzu_test]
fn quantized_matches_oracle_all_types() {
    quantized_matches_oracle::<f32>();
    quantized_matches_oracle::<bf16>();
}

/// The model's rows of multiples of 128 and raw rows of 64 and 192, whose odd ladder-index counts pack rows from the
/// middle of a byte, with and without the transform; an empty vocabulary, batch and row.
fn d4s4_matches_oracle<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let mut cases = [64, 128, 192, 256]
        .into_iter()
        .flat_map(|dim| {
            let case = Case::<T>::new(D4S4, None, (VOCAB, dim), &TOKENS, dim as usize);
            [case.clone(), case.hadamard(dim as usize)]
        })
        .collect::<Vec<_>>();
    cases.push(Case::new(D4S4, None, (0, 128), &TOKENS, 1).hadamard(1));
    cases.push(Case::new(D4S4, None, (VOCAB, 128), &[], 1).hadamard(1));
    cases.push(Case::new(D4S4, None, (VOCAB, 0), &TOKENS, 1));
    check(&fixture, &cases);
    fixture.assert_clean();
}

#[uzu_test]
fn d4s4_matches_oracle_all_types() {
    d4s4_matches_oracle::<f32>();
    d4s4_matches_oracle::<bf16>();
}

/// `count` values cycling through `bits` read as FP32 and rounded to T.
fn special<T: Float>(
    bits: &[u32],
    count: usize,
) -> Vec<T> {
    (0..count).map(|i| T::from(f32::from_bits(bits[i % bits.len()])).unwrap()).collect()
}

/// NaN, infinities, signed zeros, subnormals and the largest finite values in every table, including subnormal
/// products scaled back to normal results, exact cancellations to +0 and overflow to infinity; both transforms of
/// them; and invalid tokens over NaN tables with all-negative factors, which still give +0.
fn special_values_match_oracle<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let specials = [
        0x7fc0_0000,
        0xff80_0000,
        0x7f80_0000,
        0x0000_0000,
        0x8000_0000,
        0x0001_0000,
        0x8020_0000,
        0x0080_0000,
        0x7f7f_0000,
        0x3f80_0000,
        0xc040_0000,
        0x0c00_0000,
        0x1000_0000,
    ];
    let mut cases = Vec::new();
    for (dim, hadamard) in [(64, false), (64, true)] {
        let with = |case: Case<T>| {
            if hadamard {
                case.hadamard(3)
            } else {
                case
            }
        };
        let mut dense = with(Case::new(Dense, None, (VOCAB, dim), &TOKENS, 1));
        dense.values = bytemuck::cast_slice(&special::<T>(&specials, (VOCAB * dim) as usize)).to_vec();
        cases.push(dense.clone());
        for input_scale in [1e-39, 3e37, -0.0, f32::NAN] {
            cases.push(Case {
                input_scale,
                ..dense.clone()
            });
        }
        for (mode, method) in [(U4, ScaleSymmetric), (I8, ScaleSymmetric), (U8, ScaleZeroPoint), (U4, ScaleBias)] {
            let mut quantized = with(Case::new(Quantized, Some((mode, method, 16)), (VOCAB, dim), &TOKENS, 2));
            quantized.scales = Some(special(&specials, (VOCAB * dim / 16) as usize));
            quantized.biases = quantized.biases.map(|biases| special(&specials[3..], biases.len()));
            cases.push(quantized.clone());
            cases.push(Case {
                input_scale: 2e30,
                ..quantized
            });
        }
        let mut d4s4 = with(Case::new(D4S4, None, (VOCAB, dim), &TOKENS, 3));
        d4s4.scales = Some(special(&specials, VOCAB as usize));
        d4s4.ladder = Some(
            [f16::NAN, f16::INFINITY, f16::NEG_ZERO, f16::from_bits(1), f16::MAX, f16::ONE].repeat(3)[..16].to_vec(),
        );
        cases.push(d4s4.clone());
        cases.push(Case {
            input_scale: 1e30,
            ..d4s4
        });
    }
    for case in [
        Case::<T>::new(Dense, None, (VOCAB, 64), &TOKENS, 4),
        Case::new(Quantized, Some((U4, ScaleBias, 32)), (VOCAB, 64), &TOKENS, 5),
        Case::new(D4S4, None, (VOCAB, 64), &TOKENS, 6),
    ] {
        let nan = |values: Option<Vec<T>>| values.map(|values| vec![T::nan(); values.len()]);
        cases.push(Case {
            token_ids: vec![VOCAB, u32::MAX, VOCAB + 1],
            values: vec![0xff; case.values.len()],
            scales: nan(case.scales.clone()),
            biases: nan(case.biases.clone()),
            factors: Some(vec![-1; 64]),
            ladder: case.ladder.clone().map(|ladder| vec![f16::NAN; ladder.len()]),
            ..case
        });
    }
    check(&fixture, &cases);
    fixture.assert_clean();
}

#[uzu_test]
fn special_values_match_oracle_all_types() {
    special_values_match_oracle::<f32>();
    special_values_match_oracle::<bf16>();
}

/// Subnormal code products and offsets whose FP32 sum and difference scale back to normal results, which a device
/// flushing subnormal operands or negating a subnormal loses: U4 code 3 with scale 2^-128 (0x00200000) and input scale
/// 2^100 gives 5 2^-28 (0x32a00000) with the ScaleBias bias 2^-127 and -5 2^-28 (0xb2a00000) after the symmetric
/// midpoint 8, exactly on the oracle, the CPU and Vulkan.
#[uzu_test]
fn subnormal_offsets_scale_to_normal_results() {
    let fixture = KernelFixture::new();
    let cases = [ScaleBias, ScaleSymmetric].map(|method| {
        let mut case = Case::<f32>::new(Quantized, Some((U4, method, 32)), (1, 32), &[0], 1);
        case.values = vec![0x33; 16];
        case.scales = Some(vec![f32::from_bits(0x0020_0000)]);
        case.biases = case.biases.map(|_| vec![f32::from_bits(0x0040_0000)]);
        case.input_scale = f32::from_bits(0x7180_0000);
        case
    });
    let outputs = Case::gpu(&fixture, &cases, fixture.encoding());
    for ((case, gpu), expected) in cases.iter().zip(outputs).zip([0x32a0_0000u32, 0xb2a0_0000]) {
        let oracle = case.oracle().into_iter().map(|(_, center)| center as f32).collect::<Vec<_>>();
        for (side, values) in [("oracle", oracle), ("CPU", case.cpu()), ("Vulkan", gpu)] {
            let bits = values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits, vec![expected; 32], "{} {side}", case.label());
        }
    }
    fixture.assert_clean();
}

/// Cases where the canonical staging differs from a fused multiply-add or a reassociated product, each difference
/// asserted before CPU and Vulkan are checked against the staged oracle: a code product cancelled by a bias rounded
/// from it and by the symmetric midpoint's product (F32 only: BF16 scales times 8-bit codes are exact), and overflow and
/// underflow of a first product that later factors would undo. The shared corpora themselves need rounding: some F32
/// code products of the quantized scales and some chained D4S4 products are inexact.
fn staging_is_canonical<T: ArrayElement + Float + NoUninit + Debug>() {
    let fixture = KernelFixture::new();
    let wide = |value: &T| value.to_f64().unwrap();
    let f32_scales = T::data_type() == DataType::F32;
    let quantized = Case::<T>::new(Quantized, Some((U8, ScaleBias, 32)), (VOCAB, 64), &TOKENS, 1);
    let products = quantized
        .scales
        .as_ref()
        .unwrap()
        .iter()
        .flat_map(|scale| (0..256).map(move |code| wide(scale) * f64::from(code)));
    let inexact = products.filter(|&product| round32(product) != product).count();
    assert!(!f32_scales || inexact > 0, "the quantized corpus has only exact code products");
    let d4s4 = Case::<T>::new(D4S4, None, (VOCAB, 128), &TOKENS, 128);
    let steps =
        d4s4.scales.as_ref().unwrap().iter().flat_map(|scale| {
            d4s4.ladder.as_ref().unwrap().iter().map(move |step| round32(wide(scale) * step.to_f64()))
        });
    let points = d4s4.codebook.as_ref().unwrap();
    let inexact = steps
        .flat_map(|step| points.iter().map(move |&point| step * f64::from(point)))
        .filter(|&product| round32(product) != product)
        .count();
    assert!(inexact > 0, "the D4S4 corpus has only exact chained products");

    let scale = T::from(f32::from_bits(0x3f9d_70a4)).unwrap();
    let quantized = |method, code: u8, scale: T, input_scale: f32| {
        let mut case = Case::<T>::new(Quantized, Some((U8, method, 32)), (1, 32), &[0], 1);
        case.values = vec![code; 32];
        case.scales = Some(vec![scale]);
        case.biases = case.biases.map(|_| vec![T::from(-round32(wide(&scale) * f64::from(code))).unwrap()]);
        Case {
            input_scale,
            ..case
        }
    };
    let d4s4 = |row_scale: f32, step: f16, input_scale: f32| Case {
        scales: Some(vec![T::from(row_scale).unwrap()]),
        ladder: Some(vec![step; 16]),
        codebook: Some(vec![3; 1024]),
        input_scale,
        ..Case::<T>::new(D4S4, None, (1, 64), &[0], 1)
    };
    let (s, two) = (wide(&scale), |exponent: i32| 2f64.powi(exponent));
    let mut cases = vec![
        (quantized(ScaleBias, 200, scale, 1.0), f32_scales.then(|| s * 200.0 - round32(s * 200.0))),
        (quantized(ScaleSymmetric, 129, scale, 1.0), f32_scales.then_some(s)),
        (
            Case {
                biases: Some(vec![T::zero()]),
                ..quantized(ScaleBias, 255, T::from(two(121)).unwrap(), two(-10) as f32)
            },
            Some(round32(two(121) * round32(255.0 * two(-10)))),
        ),
        (
            d4s4(two(121) as f32, f16::MAX, two(-40) as f32),
            Some(round32(two(121) * round32(65504.0 * round32(3.0 * two(-40))))),
        ),
        (
            d4s4(two(-130) as f32, f16::from_bits(1), two(100) as f32),
            Some(round32(two(-130) * round32(two(-24) * round32(3.0 * two(100))))),
        ),
    ];
    for (case, alternative) in &mut cases {
        if let Some(alternative) = alternative {
            assert_ne!(to::<T>(*alternative), case.oracle()[0].1, "{}: insensitive to the staging", case.label());
        }
    }
    check(&fixture, &cases.into_iter().map(|(case, _)| case).collect::<Vec<_>>());
    fixture.assert_clean();
}

#[uzu_test]
fn staging_is_canonical_all_types() {
    staging_is_canonical::<f32>();
    staging_is_canonical::<bf16>();
}

/// Construction rejects F16 and I32, a zero group size and optional specializations present where absent or absent
/// where present. `encode` rejects every optional buffer missing where required or present where not, also for empty
/// batches and rows, before recording anything; the same command buffer then completes valid work and rejected calls leave
/// their buffers untouched.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    let new = |data_type, table_kind, group_size, mode, method, use_hadamard| {
        InputEmbeddingLookupVulkanKernel::new(
            &fixture.context,
            data_type,
            table_kind,
            group_size,
            mode,
            method,
            use_hadamard,
        )
    };
    for data_type in [DataType::F16, DataType::I32] {
        let result = new(data_type, Dense, None, None, None, false);
        assert!(
            matches!(
                result,
                Err(Error::KernelVariant {
                    kernel: "InputEmbeddingLookup",
                    ..
                })
            ),
            "{data_type:?}"
        );
    }
    let result = new(DataType::F32, Quantized, Some(0), Some(U8), Some(ScaleBias), false);
    assert!(
        matches!(
            result,
            Err(Error::KernelPrecondition {
                kernel: "InputEmbeddingLookup",
                ..
            })
        ),
        "group size 0"
    );
    for (table_kind, group_size, mode, method) in [
        (Dense, Some(32), None, None),
        (D4S4, None, Some(U4), None),
        (Dense, None, None, Some(ScaleBias)),
        (Quantized, None, Some(U4), Some(ScaleBias)),
        (Quantized, Some(32), None, Some(ScaleBias)),
        (Quantized, Some(32), Some(U4), None),
    ] {
        let construct = AssertUnwindSafe(|| new(DataType::F32, table_kind, group_size, mode, method, false));
        assert!(catch_unwind(construct).is_err(), "{table_kind:?} {group_size:?} {mode:?} {method:?} accepted");
    }
    let untouched = fixture.buffer(&[0u8; 4096]);
    let mut encoding = fixture.encoding();
    let kernels = [
        Case::<f32>::new(Dense, None, (VOCAB, 64), &TOKENS, 1),
        Case::new(Quantized, Some((U4, ScaleZeroPoint, 32)), (VOCAB, 64), &TOKENS, 1).hadamard(1),
        Case::new(Quantized, Some((U8, ScaleBias, 32)), (VOCAB, 64), &TOKENS, 1),
        Case::new(Quantized, Some((I8, ScaleSymmetric, 32)), (VOCAB, 64), &TOKENS, 1),
        Case::new(D4S4, None, (VOCAB, 128), &TOKENS, 1).hadamard(1),
    ];
    for case in &kernels {
        let kernel = case.vulkan_kernel(&fixture);
        let present = case.inputs().map(|input| input.is_some());
        let shapes = [(0, case.model_dim), (2, 0), (2, case.model_dim)];
        for (flip, (batch, model_dim)) in (2..present.len()).flat_map(|flip| shapes.map(|shape| (flip, shape))) {
            let mut arguments = present.map(|present| present.then_some((&untouched, 0..512)));
            arguments[flip] = arguments[flip].is_none().then_some((&untouched, 0..512));
            let [_, _, scales, zero_points, biases, factors, ladder_indices, ladder, codebook] = arguments;
            let encode = AssertUnwindSafe(|| unsafe {
                // SAFETY: never dispatched: the optional-argument assertion fails before recording.
                kernel.encode(
                    (&untouched, 0..512),
                    (&untouched, 0..512),
                    scales,
                    zero_points,
                    biases,
                    factors,
                    ladder_indices,
                    ladder,
                    codebook,
                    (&untouched, 0..512),
                    batch,
                    VOCAB,
                    model_dim,
                    1.0,
                    &mut encoding,
                );
            });
            let accepted = catch_unwind(encode).is_ok();
            assert!(!accepted, "{} argument {flip} {batch}x{model_dim} accepted", case.label());
        }
    }
    let valid = &kernels[1];
    let outputs = Case::gpu(&fixture, std::slice::from_ref(valid), encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    let untouched = unsafe { KernelFixture::read::<u8>(&untouched) };
    assert!(untouched.iter().all(|&byte| byte == 0), "a rejected call wrote");
    check_bounds(&valid.oracle(), &valid.cpu(), &outputs[0], "valid work");
    fixture.assert_clean();
}

/// The binding's preconditions on rows: even for U4, multiples of 32 with the transform and of 64 for D4S4, checked
/// also for empty batches before recording anything; the same command buffer then completes valid work and rejected
/// outputs stay untouched.
#[uzu_test]
fn rejects_violated_preconditions() {
    let fixture = KernelFixture::new();
    let invalid = [
        Case::<f32>::new(Quantized, Some((U4, ScaleBias, 8)), (VOCAB, 32), &TOKENS, 1),
        Case::new(Dense, None, (VOCAB, 32), &TOKENS, 1).hadamard(1),
        Case::new(Quantized, Some((U8, ScaleSymmetric, 8)), (VOCAB, 32), &TOKENS, 1).hadamard(1),
        Case::new(D4S4, None, (VOCAB, 64), &TOKENS, 1),
        Case::new(D4S4, None, (VOCAB, 64), &TOKENS, 1).hadamard(1),
    ];
    let rows = [31, 48, 80, 96, 96];
    let untouched = fixture.buffer(&[0u8; 4096]);
    let mut encoding = fixture.encoding();
    for (case, model_dim) in invalid.iter().zip(rows) {
        let kernel = case.vulkan_kernel(&fixture);
        for batch in [0, 2] {
            let arguments = case.inputs().map(|input| input.map(|_| (&untouched, 0..512)));
            let [_, _, scales, zero_points, biases, factors, ladder_indices, ladder, codebook] = arguments;
            let encode = AssertUnwindSafe(|| unsafe {
                // SAFETY: never dispatched: the precondition fails before recording.
                kernel.encode(
                    (&untouched, 0..512),
                    (&untouched, 0..512),
                    scales,
                    zero_points,
                    biases,
                    factors,
                    ladder_indices,
                    ladder,
                    codebook,
                    (&untouched, 0..512),
                    batch,
                    VOCAB,
                    model_dim,
                    1.0,
                    &mut encoding,
                );
            });
            assert!(catch_unwind(encode).is_err(), "{} rows {model_dim} batch {batch} accepted", case.label());
        }
    }
    let outputs = Case::gpu(&fixture, &invalid, encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatches.
    let untouched = unsafe { KernelFixture::read::<u8>(&untouched) };
    assert!(untouched.iter().all(|&byte| byte == 0), "a rejected call wrote");
    for (case, gpu) in invalid.iter().zip(outputs) {
        check_bounds(&case.oracle(), &case.cpu(), &gpu, &format!("valid {}", case.label()));
    }
    fixture.assert_clean();
}

/// Run alone: `cargo test ... input_embedding_lookup_test::throughput -- --ignored --nocapture`. Construction cost,
/// then decode and prefill batches of random tokens from model-shaped dense, U4 ScaleBias groups of 64 and D4S4 tables,
/// each with and without the output transform.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float + NoUninit + Debug>(fixture: &KernelFixture) {
        let (vocab, dim) = (4096u32, 2048u32);
        let tables = [
            (Case::<T>::new(Dense, None, (vocab, dim), &[], 1), "dense"),
            (Case::new(Quantized, Some((U4, ScaleBias, 64)), (vocab, dim), &[], 1), "U4 ScaleBias 64"),
            (Case::new(Quantized, Some((U4, ScaleBias, 64)), (vocab, dim), &[], 1).hadamard(1), "U4 ScaleBias 64"),
            (Case::new(D4S4, None, (vocab, dim), &[], 1).hadamard(1), "D4S4"),
            (Case::new(D4S4, None, (vocab, dim), &[], 1), "D4S4"),
            (Case::new(Dense, None, (vocab, dim), &[], 1).hadamard(1), "dense"),
        ];
        let mut construction = (0..11)
            .map(|_| {
                let start = Instant::now();
                tables[3].0.vulkan_kernel(fixture);
                start.elapsed()
            })
            .collect::<Vec<_>>();
        let first = construction[0];
        construction.sort();
        eprintln!(
            "InputEmbeddingLookup {:?} construction: first {first:?}, median of 11 {:?}",
            T::data_type(),
            construction[5]
        );
        for (table, name) in &tables {
            let kernel = table.vulkan_kernel(fixture);
            let inputs =
                table.inputs().map(|input| input.filter(|bytes| !bytes.is_empty()).map(|bytes| fixture.buffer(&bytes)));
            for batch in [1u32, 128, 1024] {
                let tokens = (0..batch).map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % vocab).collect::<Vec<_>>();
                let tokens = fixture.buffer(&tokens);
                let output = fixture.buffer(&vec![T::zero(); (batch * dim) as usize]);
                let [_, values, scales, zero_points, biases, factors, ladder_indices, ladder, codebook] =
                    inputs.each_ref().map(|input| input.as_ref().map(|buffer| (buffer, 0..buffer.size())));
                let (gpu, wall) = fixture.median_times(|encoding| unsafe {
                    // SAFETY: every range holds the table's canonical payload; the output does not alias them.
                    kernel.encode(
                        (&tokens, 0..u64::from(batch) * 4),
                        values.clone().unwrap(),
                        scales.clone(),
                        zero_points.clone(),
                        biases.clone(),
                        factors.clone(),
                        ladder_indices.clone(),
                        ladder.clone(),
                        codebook.clone(),
                        (&output, 0..output.size()),
                        batch,
                        vocab,
                        dim,
                        1.37,
                        encoding,
                    );
                });
                eprintln!(
                    "MEASURE InputEmbeddingLookup {:?} {name} hadamard {} {batch}x{dim}: median of 10 after 3 warm-up: GPU {gpu:?}, wall {wall:?}",
                    T::data_type(),
                    table.factors.is_some()
                );
            }
        }
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
