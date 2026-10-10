use std::{
    fmt::Debug,
    mem::size_of,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Instant,
};

use bytemuck::NoUninit;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::{kernel_fixture::KernelFixture, round32};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Kernels,
            gpu_types::{ActivationTransformOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
            kernel::{ActivationQuantization, ActivationTransformKernel, matmul::Int8CodeLayout},
        },
        cpu::{Cpu, kernel::activation_transform::quantize_transformed_row},
        vulkan::{
            ActivationTransformVulkanKernel, Error, VkBuffer, vk_kernels::TestActivationQuantizationVulkanKernel,
        },
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

const BLOCK: usize = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
const CODE_SENTINEL: i8 = 0x5a;
const SCALE_SENTINEL: f32 = -7.0;
const SUM_SENTINEL: i32 = 0x5a5a_5a5a;
const FULL_PRECISION: [ActivationTransformOp; 2] = [ActivationTransformOp::InputRht, ActivationTransformOp::OutputRht];

/// Deterministic Hadamard factors, ±1.
pub fn signs(
    count: usize,
    seed: usize,
) -> Vec<i32> {
    (0..count)
        .map(|i| {
            if (i * 2_654_435_761 + seed * 40_503) >> 9 & 1 == 0 {
                1
            } else {
                -1
            }
        })
        .collect()
}

/// Mixed-sign values in [-50, 50] scaled by 2^-4 to 2^4 per 37 elements, so groups have distinct maxima.
pub fn values<T: Float>(
    count: usize,
    seed: usize,
) -> Vec<T> {
    let value = |i: usize| ((i * 7919 + seed * 104_729) % 4001) as f32 / 40.0 - 50.0;
    (0..count).map(|i| T::from(value(i) * 2f32.powi((i / 37 % 9) as i32 - 4)).unwrap()).collect()
}

/// Every model-admitted quantization: scale and weight groups of 32, 64 or 128 with the weight group at most the scale
/// group, with and without group sums, in both code layouts.
pub fn quantizations() -> Vec<ActivationQuantization> {
    let groups = [32, 64, 128];
    let layouts = [Int8CodeLayout::Sequential, Int8CodeLayout::GroupedByNibble];
    let settings = groups.iter().flat_map(|&scale| groups.iter().map(move |&weight| (scale, weight)));
    let settings = settings.flat_map(|(scale, weight)| [false, true].map(|sums| (scale, weight, sums)));
    let settings = settings.flat_map(|(scale, weight, sums)| layouts.map(|layout| (scale, weight, sums, layout)));
    settings
        .filter_map(|(scale, weight, sums, layout)| ActivationQuantization::new(scale, weight, sums, layout))
        .collect()
}

/// A model quantization as raw `(scale group, sum group, code layout)` settings.
pub fn raw(quantization: ActivationQuantization) -> (usize, Option<usize>, Int8CodeLayout) {
    let sum_group = quantization.sum_group_size().map(|group| group as usize);
    (quantization.scale_group_size() as usize, sum_group, quantization.code_layout())
}

/// Raw quantization settings for messages.
pub fn label((scale_group, sum_group, layout): (usize, Option<usize>, Int8CodeLayout)) -> String {
    format!("scale group {scale_group} sum group {sum_group:?} grouped by nibble {}", layout.is_grouped_by_nibble())
}

/// The canonical CPU quantization of FP32 rows of `count` elements: codes, scales and group sums, empty when absent.
pub fn quantized(
    transformed: &[f32],
    count: usize,
    (scale_group, sum_group, layout): (usize, Option<usize>, Int8CodeLayout),
) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    assert!(count > 0 && transformed.len().is_multiple_of(count), "rows of {count} elements");
    let mut codes = vec![0; transformed.len()];
    let mut scales = vec![0.0; transformed.len() / scale_group];
    let mut sums = vec![0; sum_group.map_or(0, |group| transformed.len() / group)];
    for (row, values) in transformed.chunks_exact(count).enumerate() {
        let row_sums = sum_group.map(|group| count / group);
        quantize_transformed_row(
            values,
            scale_group,
            sum_group,
            &mut codes[row * count..][..count],
            &mut scales[row * count / scale_group..][..count / scale_group],
            row_sums.map(|row_sums| &mut sums[row * row_sums..][..row_sums]),
            layout,
        );
    }
    (codes, scales, sums)
}

/// Asserts quantization outputs are exactly `expected`: lengths, scales bit for bit, codes and sums, printing the first
/// difference of each with the logical input it came from.
pub fn assert_quantized(
    actual: &(Vec<i8>, Vec<f32>, Vec<i32>),
    expected: &(Vec<i8>, Vec<f32>, Vec<i32>),
    inputs: &[f32],
    case: &str,
) {
    let lengths = |outputs: &(Vec<i8>, Vec<f32>, Vec<i32>)| [outputs.0.len(), outputs.1.len(), outputs.2.len()];
    assert_eq!(lengths(actual), lengths(expected), "{case}: output lengths");
    assert_eq!(actual.0.len(), inputs.len(), "{case}: one code per input");
    let scale = actual.1.iter().zip(&expected.1).position(|(actual, expected)| actual.to_bits() != expected.to_bits());
    let scales_per_input = actual.1.len() as f64 / inputs.len() as f64;
    assert!(
        scale.is_none(),
        "{case}: scale {scale:?} differs: {:?} vs expected {:?}, scales per input {scales_per_input}",
        scale.map(|i| actual.1[i].to_bits()),
        scale.map(|i| expected.1[i].to_bits())
    );
    let code = actual.0.iter().zip(&expected.0).position(|(actual, expected)| actual != expected);
    assert!(
        code.is_none(),
        "{case}: stored code {code:?} differs: {:?} vs expected {:?}",
        code.map(|i| actual.0[i]),
        code.map(|i| expected.0[i])
    );
    assert!(actual.2 == expected.2, "{case}: group sums differ");
}

/// `((lo, hi), center)` times a sign, exactly.
fn signed(
    ((lo, hi), center): ((f64, f64), f64),
    sign: f64,
) -> ((f64, f64), f64) {
    match sign < 0.0 {
        true => ((-hi, -lo), -center),
        false => ((lo, hi), center),
    }
}

/// The staged FP32 transform of rows of `factors.len()` bounded FP32 values `((lo, hi), center)` in FP64, each stage
/// rounded to FP32 (subnormals kept): the factored values (before the transform for the input transform, after it for
/// the output one), the butterfly stages of each block of `block` values, then the product with the CPU's FP32
/// reciprocal of sqrt(block). Every stage is exact or a single correctly rounded, monotonic operation, so bounds
/// propagate: sums add like bounds, differences opposite ones. Exact inputs give one value as degenerate bounds.
pub fn transform_oracle(
    rows: &[((f64, f64), f64)],
    factors: &[i32],
    input_rht: bool,
    block: usize,
) -> Vec<((f64, f64), f64)> {
    assert!(factors.iter().all(|factor| factor.abs() == 1), "factors are signs");
    assert!(rows.len().is_multiple_of(factors.len().max(1)), "whole rows");
    let reciprocal = f64::from(1.0 / (block as f32).sqrt());
    let mut result = Vec::with_capacity(rows.len());
    for row in rows.chunks_exact(factors.len().max(1)) {
        for (values, signs) in row.chunks_exact(block).zip(factors.chunks_exact(block)) {
            let mut values = values.to_vec();
            if input_rht {
                values.iter_mut().zip(signs).for_each(|(value, &sign)| *value = signed(*value, f64::from(sign)));
            }
            let mut stride = 1;
            while stride < block {
                for lane in (0..block).filter(|lane| lane & stride == 0) {
                    let (((a_lo, a_hi), a), ((b_lo, b_hi), b)) = (values[lane], values[lane | stride]);
                    values[lane] = ((round32(a_lo + b_lo), round32(a_hi + b_hi)), round32(a + b));
                    values[lane | stride] = ((round32(a_lo - b_hi), round32(a_hi - b_lo)), round32(a - b));
                }
                stride <<= 1;
            }
            for (((lo, hi), center), &sign) in values.into_iter().zip(signs) {
                let scaled = |value: f64| round32(value * reciprocal);
                let sign = if input_rht {
                    1.0
                } else {
                    f64::from(sign)
                };
                result.push(signed(((scaled(lo), scaled(hi)), scaled(center)), sign));
            }
        }
    }
    result
}

/// `value` rounded to `T`, monotonic.
pub fn to<T: Float>(value: f64) -> f64 {
    T::from(value).unwrap().to_f64().unwrap()
}

/// Checks CPU and Vulkan outputs against bounds already rounded to `T`: NaN exactly where the oracle is NaN, its
/// infinities exactly, its zero sign where both bounds are zeros of one sign, otherwise within bounds, which must be
/// finite for finite results. Returns the number of Vulkan results differing from the CPU's and the most storage steps.
pub fn check_bounds<T: Float + NoUninit + Debug>(
    bounds: &[((f64, f64), f64)],
    cpu: &[T],
    gpu: &[T],
    case: &str,
) -> [i64; 2] {
    assert_eq!((cpu.len(), gpu.len()), (bounds.len(), bounds.len()), "{case}: output lengths");
    let mut violations = 0;
    let mut difference = [0i64; 2];
    for (index, ((&((lo, hi), center), &cpu), &gpu)) in bounds.iter().zip(cpu).zip(gpu).enumerate() {
        assert!(!center.is_finite() || (lo.is_finite() && hi.is_finite()), "{case}: element {index}: unbounded oracle");
        for (side, value) in [("Vulkan", gpu), ("CPU", cpu)] {
            let value = value.to_f64().unwrap();
            let valid = match center.is_finite() {
                false => center.is_nan() && value.is_nan() || center == value,
                true if value == 0.0 && lo == 0.0 && hi == 0.0 && lo.is_sign_negative() == hi.is_sign_negative() => {
                    value.is_sign_negative() == center.is_sign_negative()
                },
                true => lo <= value && value <= hi,
            };
            if !valid {
                violations += 1;
                if violations <= 5 {
                    eprintln!("{case}: element {index}: {side} {value:e}, oracle {center:e} in [{lo:e}, {hi:e}]");
                }
            }
        }
        if bytemuck::bytes_of(&cpu) != bytemuck::bytes_of(&gpu) && !(cpu.is_nan() && gpu.is_nan()) {
            let steps = (KernelFixture::ordinal(cpu) - KernelFixture::ordinal(gpu)).abs();
            difference = [difference[0] + 1, difference[1].max(steps)];
        }
    }
    assert_eq!(violations, 0, "{case}: {violations} results outside the oracle bounds");
    difference
}

/// The specialization of raw quantization settings as the model constructor makes it: without quantization groups of
/// 32, without sums a sum group of 32.
fn settings(quantization: Option<(usize, Option<usize>, Int8CodeLayout)>) -> (bool, u32, u32) {
    quantization.map_or((false, 32, 32), |(scale_group, sum_group, layout)| {
        let sum_group = sum_group.map_or(HADAMARD_TRANSFORM_BLOCK_SIZE, |group| group as u32);
        (layout.is_grouped_by_nibble(), scale_group as u32, sum_group)
    })
}

fn ops(
    full_precision: ActivationTransformOp,
    quantization: Option<(usize, Option<usize>, Int8CodeLayout)>,
) -> ActivationTransformOp {
    match quantization {
        None => full_precision,
        Some((_, Some(_), _)) => ActivationTransformOp::QuantizeWithGroupSums,
        Some(_) => ActivationTransformOp::Quantize,
    }
}

/// The CPU kernel through the shared trait: full precision, codes, scales and group sums, each empty when absent. CPU
/// buffers cannot be empty.
pub fn cpu_outputs<T: ArrayElement + Float, BiasT: ArrayElement + Float>(
    (op, in_place, bias, quantization, input, factors, batch): (
        ActivationTransformOp,
        bool,
        Option<&[BiasT]>,
        Option<(usize, Option<usize>, Int8CodeLayout)>,
        &[T],
        &[i32],
        usize,
    )
) -> (Vec<T>, Vec<i8>, Vec<f32>, Vec<i32>) {
    let mut result = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    if input.is_empty() {
        return result;
    }
    let context = create_context::<Cpu>();
    let (grouped, scale_group, sum_group) = settings(quantization);
    let ops = ops(op, quantization);
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::ActivationTransformKernel::new(
        &context,
        T::data_type(),
        BiasT::data_type(),
        ops,
        grouped,
        in_place,
        scale_group,
        sum_group,
        bias.is_some(),
    )
    .expect("CPU ActivationTransform");
    let buffer = |values: &[T]| create_buffer_with_data::<Cpu, T>(&context, values);
    let input_buffer = (!in_place).then(|| buffer(input));
    let mut fp = quantization.is_none().then(|| buffer(input));
    let bias = bias.map(|bias| create_buffer_with_data::<Cpu, BiasT>(&context, bias));
    let n = input.len();
    let mut codes = quantization.map(|_| create_buffer_with_data::<Cpu, i8>(&context, &vec![0; n]));
    let mut scales =
        quantization.map(|_| create_buffer_with_data::<Cpu, f32>(&context, &vec![0.0; n / scale_group as usize]));
    let sums = quantization.and_then(|(_, sum_group, _)| sum_group).map(|group| n / group);
    let mut sums = sums.map(|sums| create_buffer_with_data::<Cpu, i32>(&context, &vec![0; sums]));
    let factors = create_buffer_with_data::<Cpu, i32>(&context, factors);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    kernel.encode(
        input_buffer.as_ref(),
        fp.as_mut(),
        bias.as_ref(),
        codes.as_mut(),
        scales.as_mut(),
        sums.as_mut(),
        &factors,
        batch as u32,
        (n / batch) as u32,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    if let Some(fp) = &fp {
        result.0 = buffer_to_vec::<Cpu, T>(fp);
    }
    if let (Some(codes), Some(scales)) = (&codes, &scales) {
        (result.1, result.2) = (buffer_to_vec::<Cpu, i8>(codes), buffer_to_vec::<Cpu, f32>(scales));
    }
    if let Some(sums) = &sums {
        result.3 = buffer_to_vec::<Cpu, i32>(sums);
    }
    result
}

fn vulkan_kernel<T: ArrayElement, BiasT: ArrayElement>(
    fixture: &KernelFixture,
    op: ActivationTransformOp,
    in_place: bool,
    has_bias: bool,
    quantization: Option<(usize, Option<usize>, Int8CodeLayout)>,
) -> ActivationTransformVulkanKernel {
    let (grouped, scale_group, sum_group) = settings(quantization);
    let ops = ops(op, quantization);
    let (t, bias_t) = (T::data_type(), BiasT::data_type());
    ActivationTransformVulkanKernel::new(
        &fixture.context,
        t,
        bias_t,
        ops,
        grouped,
        in_place,
        scale_group,
        sum_group,
        has_bias,
    )
    .expect("Vulkan ActivationTransform")
}

/// Records every case into one command buffer over guarded ranges, then returns each case's outputs as `cpu_outputs`
/// after checking every guard and that read-only inputs are unchanged.
pub fn gpu_outputs<T: ArrayElement + Float, BiasT: ArrayElement + Float>(
    fixture: &KernelFixture,
    cases: &[(
        ActivationTransformOp,
        bool,
        Option<&[BiasT]>,
        Option<(usize, Option<usize>, Int8CodeLayout)>,
        &[T],
        &[i32],
        usize,
    )],
) -> Vec<(Vec<T>, Vec<i8>, Vec<f32>, Vec<i32>)> {
    let (sentinel, bias_sentinel) = (T::from(-7.0).unwrap(), BiasT::from(-7.0).unwrap());
    fn range(guarded: &Option<(Arc<VkBuffer>, Range<u64>)>) -> Option<(&Arc<VkBuffer>, Range<u64>)> {
        guarded.as_ref().map(|(buffer, range)| (buffer, range.clone()))
    }
    let buffers = cases
        .iter()
        .map(|&(op, in_place, bias, quantization, input, factors, batch)| {
            let n = input.len();
            let scale_group = quantization.map_or(1, |(scale_group, ..)| scale_group);
            let sum_group = quantization.and_then(|(_, sum_group, _)| sum_group);
            (
                vulkan_kernel::<T, BiasT>(fixture, op, in_place, bias.is_some(), quantization),
                (!in_place).then(|| fixture.guarded(input, sentinel)),
                quantization.is_none().then(|| {
                    fixture.guarded(
                        &if in_place {
                            input.to_vec()
                        } else {
                            vec![sentinel; n]
                        },
                        sentinel,
                    )
                }),
                bias.map(|bias| fixture.guarded(bias, bias_sentinel)),
                quantization.map(|_| fixture.guarded(&vec![CODE_SENTINEL; n], CODE_SENTINEL)),
                quantization.map(|_| fixture.guarded(&vec![SCALE_SENTINEL; n / scale_group], SCALE_SENTINEL)),
                sum_group.map(|group| fixture.guarded(&vec![SUM_SENTINEL; n / group], SUM_SENTINEL)),
                fixture.guarded(factors, SUM_SENTINEL),
                (n.checked_div(batch).unwrap_or(0), batch),
            )
        })
        .collect::<Vec<_>>();
    let mut encoding = fixture.encoding();
    for (kernel, input, fp, bias, codes, scales, sums, factors, (count, batch)) in &buffers {
        // SAFETY: each range holds exactly the elements its argument covers for `batch` rows of `count` elements, and
        // only in-place full precision aliases, the output with itself.
        unsafe {
            kernel.encode(
                range(input),
                range(fp),
                range(bias),
                range(codes),
                range(scales),
                range(sums),
                (&factors.0, factors.1.clone()),
                *batch as u32,
                *count as u32,
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    cases
        .iter()
        .zip(&buffers)
        .map(
            |(
                &(_, _, bias_values, _, input_values, factor_values, _),
                (_, input, fp, bias, codes, scales, sums, factors, _),
            )| {
                // SAFETY: the only command buffer using these buffers has completed.
                unsafe {
                    if let Some(input) = input {
                        KernelFixture::assert_unchanged(input, sentinel, input_values, "input");
                    }
                    if let (Some(bias), Some(values)) = (bias, bias_values) {
                        KernelFixture::assert_unchanged(bias, bias_sentinel, values, "bias");
                    }
                    KernelFixture::assert_unchanged(factors, SUM_SENTINEL, factor_values, "factors");
                    (
                        fp.as_ref().map_or(Vec::new(), |fp| KernelFixture::read_guarded(fp, sentinel)),
                        codes.as_ref().map_or(Vec::new(), |codes| KernelFixture::read_guarded(codes, CODE_SENTINEL)),
                        scales
                            .as_ref()
                            .map_or(Vec::new(), |scales| KernelFixture::read_guarded(scales, SCALE_SENTINEL)),
                        sums.as_ref().map_or(Vec::new(), |sums| KernelFixture::read_guarded(sums, SUM_SENTINEL)),
                    )
                }
            },
        )
        .collect()
}

/// The FP32 input transform Vulkan computes for rows of `values` with `factors`, as the F32 kernel's exact output.
pub fn gpu_transformed(
    fixture: &KernelFixture,
    values: &[f32],
    factors: &[i32],
    batch: usize,
) -> Vec<f32> {
    gpu_outputs::<f32, f32>(fixture, &[(ActivationTransformOp::InputRht, false, None, None, values, factors, batch)])
        .remove(0)
        .0
}

/// Full-precision bounds of rows of `input`: the transform oracle rounded to `T`, then with a bias the sum with it in
/// FP32 rounded to `T`.
fn full_precision_bounds<T: Float, BiasT: Float>(
    input: &[T],
    bias: Option<&[BiasT]>,
    factors: &[i32],
    input_rht: bool,
) -> Vec<((f64, f64), f64)> {
    let rows =
        input.iter().map(|value| value.to_f64().unwrap()).map(|value| ((value, value), value)).collect::<Vec<_>>();
    let oracle = transform_oracle(&rows, factors, input_rht, BLOCK);
    let stage = |index: usize, value: f64| match bias {
        Some(bias) => to::<T>(round32(to::<T>(value) + bias[index % factors.len()].to_f64().unwrap())),
        None => to::<T>(value),
    };
    oracle
        .into_iter()
        .enumerate()
        .map(|(i, ((lo, hi), center))| ((stage(i, lo), stage(i, hi)), stage(i, center)))
        .collect()
}

/// Every full-precision setting (both transforms, both layouts, with and without bias) of every shape, the settings of
/// a shape interleaved in one command buffer: guarded spans, read-only inputs, results within the oracle bounds.
fn full_precision_matches_oracle<T: ArrayElement + Float + Debug, BiasT: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    for (batch, count) in [(0, 32), (1, 0), (1, 32), (3, 64), (3, 96), (128, 128), (1, 160), (3, 4096)] {
        let (input, factors) = (values::<T>(batch * count, count), signs(count, batch));
        let bias = values::<BiasT>(count, 7);
        let settings = FULL_PRECISION.iter().flat_map(|&op| [false, true].map(|in_place| (op, in_place)));
        let settings = settings.flat_map(|(op, in_place)| [None, Some(&bias[..])].map(|bias| (op, in_place, bias)));
        let cases = settings.map(|(op, in_place, bias)| (op, in_place, bias, None, &input[..], &factors[..], batch));
        let cases = cases.collect::<Vec<_>>();
        let outputs = gpu_outputs(&fixture, &cases);
        assert_eq!(outputs.len(), cases.len(), "one output per case");
        for (case, gpu) in cases.iter().zip(outputs) {
            let label = format!(
                "{:?}/{:?} {:?} in_place {} bias {}",
                T::data_type(),
                BiasT::data_type(),
                case.0,
                case.1,
                case.2.is_some()
            );
            let bounds = full_precision_bounds(&input, case.2, &factors, case.0 == ActivationTransformOp::InputRht);
            check_bounds(&bounds, &cpu_outputs(*case).0, &gpu.0, &format!("{label} {batch}x{count}"));
        }
    }
    fixture.assert_clean();
}

/// Inputs at and below FP32's normal range, which Vulkan FP32 arithmetic may flush, through both transforms and a
/// quantization, F32 and BF16, exactly the staged oracle and the CPU kernel: equal subnormals summing to a normal result
/// (0x00200000 32 times gives 0x00b504f3), subnormals that cancel or stay subnormal, both sides of the 2^-101 threshold of
/// the exact sum and the 2^-123 one of the exact final product, signed zeros, and subnormals next to infinities and NaNs.
#[uzu_test]
fn subnormal_transforms_match_cpu_exactly() {
    fn check<T: ArrayElement + Float + Debug>(
        fixture: &KernelFixture,
        rows: &[Vec<u32>],
    ) {
        let input = rows.concat().into_iter().map(|bits| T::from(f32::from_bits(bits)).unwrap()).collect::<Vec<_>>();
        let count = rows[0].len();
        let factor_sets = [vec![1; count], signs(count, 9)];
        let quantization = ActivationQuantization::new(64, 32, true, Int8CodeLayout::GroupedByNibble).map(raw);
        for factors in &factor_sets {
            let cases = [
                (ActivationTransformOp::InputRht, false, None, None, &input[..], &factors[..], rows.len()),
                (ActivationTransformOp::OutputRht, false, None, None, &input[..], &factors[..], rows.len()),
                (
                    ActivationTransformOp::InputRht,
                    false,
                    None::<&[T]>,
                    quantization,
                    &input[..],
                    &factors[..],
                    rows.len(),
                ),
            ];
            let outputs = gpu_outputs(fixture, &cases);
            assert_eq!(outputs.len(), cases.len(), "one output per case");
            for (case, gpu) in cases.iter().zip(outputs) {
                let label = format!("{:?} {:?} quantized {}", T::data_type(), case.0, case.3.is_some());
                let cpu = cpu_outputs(*case);
                match case.3 {
                    None => {
                        let bounds = full_precision_bounds::<T, T>(
                            &input,
                            None,
                            factors,
                            case.0 == ActivationTransformOp::InputRht,
                        );
                        check_bounds(&bounds, &cpu.0, &gpu.0, &label);
                    },
                    Some(_) => {
                        let widened = input.iter().map(|value| value.to_f32().unwrap()).collect::<Vec<_>>();
                        assert_quantized(&(gpu.1, gpu.2, gpu.3), &(cpu.1, cpu.2, cpu.3), &widened, &label);
                    },
                }
            }
        }
    }
    let fixture = KernelFixture::new();
    let block = |pattern: &[u32]| (0..BLOCK).map(|i| pattern[i % pattern.len()]).collect::<Vec<_>>();
    let rows = [
        [
            block(&[0x0020_0000]),
            block(&[0x0000_0001, 0x8000_0001]),
            block(&[0x0080_0000, 0x807f_ffff]),
            block(&[0x0000_0003]),
        ],
        [
            block(&[0x0cff_ffff, 0x0000_0005]),
            block(&[0x0d00_0000, 0x0000_0005]),
            block(&[0x0d00_0000, 0x8cff_ffff]),
            block(&[0x0000_0000, 0x8000_0000]),
        ],
        [
            block(&[0x01ff_ffff, 0x0007_ffff]),
            block(&[0x0200_0000, 0x8000_0010]),
            block(&[0x7f80_0000, 0x0000_0040]),
            block(&[0x7fc0_0000, 0x0000_0040]),
        ],
    ]
    .map(|blocks| blocks.concat());
    check::<f32>(&fixture, &rows);
    let equal = vec![f32::from_bits(0x0020_0000); BLOCK];
    let transformed = gpu_transformed(&fixture, &equal, &[1; BLOCK], 1);
    assert_eq!(transformed[0].to_bits(), 0x00b5_04f3, "32 subnormals 0x00200000 transform to a normal");
    // BF16 keeps the upper halves: subnormals from 0x0001 up, the smallest normal and their mixes with ±.
    let bf16_rows = [
        [
            block(&[0x0020_0000, 0x0001_0000]),
            block(&[0x8001_0000, 0x0040_0000]),
            block(&[0x0080_0000, 0x807f_0000]),
            block(&[0x0d00_0000, 0x0001_0000]),
        ],
        [
            block(&[0x0001_0000]),
            block(&[0x8000_0000, 0x0000_0000]),
            block(&[0x7f80_0000, 0x0002_0000]),
            block(&[0x0300_0000, 0x0005_0000]),
        ],
    ]
    .map(|blocks| blocks.concat());
    check::<bf16>(&fixture, &bf16_rows);
    fixture.assert_clean();
}

#[uzu_test]
fn f32_input_transform() {
    let fixture = KernelFixture::new();
    let (input, factors) = (values::<f32>(3 * 64, 1), signs(64, 1));
    let case = (ActivationTransformOp::InputRht, false, None::<&[f32]>, None, &input[..], &factors[..], 3);
    let bounds = full_precision_bounds::<f32, f32>(&input, None, &factors, true);
    check_bounds(&bounds, &cpu_outputs(case).0, &gpu_outputs(&fixture, &[case])[0].0, "F32 InputRht");
    fixture.assert_clean();
}

#[uzu_test]
fn full_precision_matches_oracle_all_types() {
    full_precision_matches_oracle::<f32, f32>();
    full_precision_matches_oracle::<f32, bf16>();
    full_precision_matches_oracle::<bf16, f32>();
    full_precision_matches_oracle::<bf16, bf16>();
}

/// Every model-admitted quantization, and focused smaller raw power-of-two groups, of every shape the groups divide,
/// interleaved in one command buffer per shape, one case binding an unused bias. The FP32 prequantization stage, the
/// transform of the inputs widened from `T` exactly as the kernel loads them, is checked on both sides against the
/// staged oracle; Vulkan's codes, scales and sums are exactly the canonical quantization of Vulkan's stage, and exactly
/// the CPU kernel's.
fn quantization_matches<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let (sequential, grouped) = (Int8CodeLayout::Sequential, Int8CodeLayout::GroupedByNibble);
    let small = [(2, Some(4), sequential), (16, Some(1), grouped), (128, Some(8), sequential), (1, None, grouped)];
    let settings = quantizations().into_iter().map(raw).chain(small).collect::<Vec<_>>();
    for (batch, count) in [(1, 0), (0, 128), (1, 96), (3, 160), (1, 128), (3, 384), (128, 4096)] {
        let (input, factors, bias) =
            (values::<T>(batch * count, count + 1), signs(count, batch + 1), values::<T>(count, 3));
        let divides = |(scale_group, sum_group, _): &(usize, Option<usize>, Int8CodeLayout)| {
            count % scale_group == 0 && count % sum_group.unwrap_or(1) == 0
        };
        let cases = settings
            .iter()
            .filter(|setting| divides(setting))
            .enumerate()
            .map(|(i, &q)| {
                (
                    ActivationTransformOp::InputRht,
                    false,
                    (i == 0).then_some(&bias[..]),
                    Some(q),
                    &input[..],
                    &factors[..],
                    batch,
                )
            })
            .collect::<Vec<_>>();
        let outputs = gpu_outputs(&fixture, &cases);
        assert_eq!(outputs.len(), cases.len(), "one output per case");
        if batch * count == 0 {
            assert!(outputs.iter().all(|output| output.1.is_empty() && output.2.is_empty() && output.3.is_empty()));
            continue;
        }
        let widened = input.iter().map(|value| value.to_f32().unwrap()).collect::<Vec<_>>();
        let stage = (ActivationTransformOp::InputRht, false, None::<&[f32]>, None, &widened[..], &factors[..], batch);
        let (gpu_stage, cpu_stage) = (gpu_outputs(&fixture, &[stage]).remove(0).0, cpu_outputs(stage).0);
        let bounds = full_precision_bounds::<f32, f32>(&widened, None, &factors, true);
        check_bounds(&bounds, &cpu_stage, &gpu_stage, &format!("{:?} {batch}x{count} stage", T::data_type()));
        for (case, gpu) in cases.iter().zip(outputs) {
            let label = format!("{:?} {batch}x{count} {}", T::data_type(), label(case.3.unwrap()));
            let gpu = (gpu.1, gpu.2, gpu.3);
            assert_quantized(&gpu, &quantized(&gpu_stage, count, case.3.unwrap()), &gpu_stage, &label);
            let cpu = cpu_outputs(*case);
            assert_quantized(&gpu, &(cpu.1, cpu.2, cpu.3), &gpu_stage, &format!("{label} vs CPU kernel"));
        }
    }
    fixture.assert_clean();
}

#[uzu_test]
fn quantization_matches_all_types() {
    quantization_matches::<f32>();
    quantization_matches::<bf16>();
}

/// Inputs whose input transform under +1 factors is exactly `target` in every lane: one value at the start of a block,
/// found among the 33 FP32 values around `target` sqrt(32) by transforming them on the device.
fn one_hot_inputs(
    fixture: &KernelFixture,
    targets: &[f32],
) -> Vec<f32> {
    let root = (BLOCK as f32).sqrt();
    let candidates = targets
        .iter()
        .flat_map(|&target| (-16..=16).map(move |k| f32::from_bits(((target * root).to_bits() as i32 + k) as u32)))
        .collect::<Vec<_>>();
    let mut input = vec![0.0; candidates.len() * BLOCK];
    candidates.iter().enumerate().for_each(|(block, &candidate)| input[block * BLOCK] = candidate);
    let transformed = gpu_transformed(fixture, &input, &vec![1; input.len()], 1);
    let exact = |block: usize, target: f32| transformed[block * BLOCK..][..BLOCK].iter().all(|&value| value == target);
    let found =
        targets.iter().enumerate().map(|(t, &target)| (t * 33..(t + 1) * 33).find(|&block| exact(block, target)));
    found
        .zip(targets)
        .map(|(block, target)| candidates[block.unwrap_or_else(|| panic!("no input transforms to {target:e}"))])
        .collect()
}

/// Rows of constructed groups whose transforms hit exact quantization boundaries, for scale groups of 64 and 128: an
/// infinity in the first block makes the scale exactly 1, so the other blocks' transforms (one value per block, on varied
/// lanes, so both signs occur) are the quotients themselves: midpoints k + 1/2, their FP32 neighbors, and saturation.
/// Division by sqrt(32) skips FP32 values with significands from sqrt(2) to 2, so targets lie below sqrt(2) in their
/// binade (1.5 or nextdown(0.5) are no transform's value). A first block transforming to 100.5 exercises the divided
/// scale. Zero groups, NaN groups, NaN next to finite values, and a negative infinity. Vulkan's codes, scales and sums are
/// exactly the canonical quantization of its FP32 transform, which hits every target, and midpoints round away from zero.
#[uzu_test]
fn quantizer_rounds_constructed_values() {
    let fixture = KernelFixture::new();
    let midpoints = [2, 4, 5, 8, 16, 21, 32, 44, 64, 90].map(|k| k as f32 + 0.5);
    let near = midpoints.iter().flat_map(|&m| [f32::from_bits(m.to_bits() - 1), m, f32::from_bits(m.to_bits() + 1)]);
    let unit = near.chain([0.5, 300.0, 2f32.powi(100)]).collect::<Vec<_>>();
    let scaled = unit.iter().copied().filter(|&target| target < 100.5).collect::<Vec<_>>();
    let targets = [&unit[..], &[100.5]].concat();
    let inputs = one_hot_inputs(&fixture, &targets);
    let input_of = |target: f32| inputs[targets.iter().position(|&t| t == target).unwrap()];
    for scale_group in [64, 128] {
        let blocks_per_group = scale_group / BLOCK;
        let block = |value: f32, lane: usize| {
            (0..BLOCK)
                .map(|i| {
                    if i == lane {
                        value
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        };
        let mut groups = Vec::new();
        for (leader, followers) in [(f32::INFINITY, &unit), (input_of(100.5), &scaled), (f32::NEG_INFINITY, &unit)] {
            for chunk in followers.chunks(blocks_per_group - 1) {
                let followers =
                    chunk.iter().enumerate().map(|(i, &target)| block(input_of(target), (i * 7 + 3) % BLOCK));
                groups.push([vec![block(leader, 0)], followers.collect()].concat().concat());
            }
        }
        groups.push(vec![0.0; scale_group]);
        groups.push([block(f32::NAN, 5), vec![0.0; scale_group - BLOCK]].concat());
        groups.push([block(f32::NAN, 0), block(input_of(64.5), 9), vec![0.0; scale_group - 2 * BLOCK]].concat());
        let mut row = groups
            .into_iter()
            .map(|group| [group.clone(), vec![0.0; scale_group - group.len()]].concat())
            .collect::<Vec<_>>()
            .concat();
        row.resize(row.len().div_ceil(128) * 128, 0.0);
        let factors = signs(row.len(), scale_group);
        let transform = gpu_transformed(&fixture, &row, &factors, 1);
        for target in &targets {
            assert!(transform.iter().any(|value| value.abs() == *target), "transform misses {target:e}");
        }
        let quantizations = quantizations().into_iter().filter(|q| q.scale_group_size() as usize == scale_group);
        let cases = quantizations
            .map(|q| (ActivationTransformOp::InputRht, false, None::<&[f32]>, Some(raw(q)), &row[..], &factors[..], 1));
        let cases = cases.collect::<Vec<_>>();
        for (case, gpu) in cases.iter().zip(gpu_outputs(&fixture, &cases)) {
            let quantization = case.3.unwrap();
            let expected = quantized(&transform, row.len(), quantization);
            assert_quantized(&(gpu.1, gpu.2, gpu.3), &expected, &transform, &format!("groups {scale_group}"));
            // Midpoints round away from zero, their neighbors to nearest; saturation at 127.
            let rounded = [(0.5, 1), (2.5, 3), (2.4999998, 2), (2.5000002, 3), (90.5, 91), (300.0, 127)];
            let code = |logical: usize| expected.0[quantization.2.index(logical)];
            let mut checked = 0;
            for (index, &value) in transform.iter().enumerate() {
                let magnitude = rounded.iter().find(|&&(magnitude, _)| magnitude == value.abs()).map(|&(_, code)| code);
                if let (Some(magnitude), 1.0) = (magnitude, expected.1[index / scale_group]) {
                    assert_eq!(i32::from(code(index)), magnitude * value.signum() as i32, "{value:e} at scale 1");
                    checked += 1;
                }
            }
            assert!(checked >= 2 * rounded.len() * BLOCK, "only {checked} rounding cases at scale 1");
        }
    }
    fixture.assert_clean();
}

/// Quantization settings for the direct quantizer: every model pair, and smaller raw power-of-two groups.
fn direct_settings() -> Vec<(usize, Option<usize>, Int8CodeLayout)> {
    let small = [(1, Some(1)), (2, Some(8)), (16, Some(4)), (128, Some(1)), (8, None), (4, Some(128))];
    let layouts = [Int8CodeLayout::Sequential, Int8CodeLayout::GroupedByNibble];
    let small = small.into_iter().flat_map(|(scale, sum)| layouts.map(|layout| (scale, sum, layout)));
    quantizations().into_iter().map(raw).chain(small).collect()
}

/// The production quantizer through its test entry on rows of `count` elements, every case in one command buffer over
/// guarded ranges: codes, scales and sums after checking the guards and the unchanged inputs.
fn direct_outputs(
    fixture: &KernelFixture,
    cases: &[(&[f32], usize, (usize, Option<usize>, Int8CodeLayout))],
) -> Vec<(Vec<i8>, Vec<f32>, Vec<i32>)> {
    let buffers = cases
        .iter()
        .map(|&(values, count, (scale_group, sum_group, layout))| {
            let kernel = TestActivationQuantizationVulkanKernel::new(
                &fixture.context,
                layout.is_grouped_by_nibble(),
                scale_group as u32,
                sum_group.unwrap_or(scale_group) as u32,
                sum_group.is_some(),
            )
            .expect("TestActivationQuantization");
            let n = values.len();
            (
                kernel,
                fixture.guarded(values, f32::NAN),
                fixture.guarded(&vec![CODE_SENTINEL; n], CODE_SENTINEL),
                fixture.guarded(&vec![SCALE_SENTINEL; n / scale_group], SCALE_SENTINEL),
                sum_group.map(|group| fixture.guarded(&vec![SUM_SENTINEL; n / group], SUM_SENTINEL)),
                (n / count) as u32,
                count as u32,
            )
        })
        .collect::<Vec<_>>();
    let mut encoding = fixture.encoding();
    for (kernel, values, codes, scales, sums, batch, count) in &buffers {
        // SAFETY: the ranges hold `batch` rows of `count` values, their codes, scales and sums, and do not alias.
        unsafe {
            kernel.encode(
                (&values.0, values.1.clone()),
                (&codes.0, codes.1.clone()),
                (&scales.0, scales.1.clone()),
                sums.as_ref().map(|(buffer, range)| (buffer, range.clone())),
                *batch,
                *count,
                &mut encoding,
            );
        }
    }
    KernelFixture::complete(encoding);
    cases
        .iter()
        .zip(&buffers)
        .map(|(&(inputs, ..), (_, values, codes, scales, sums, ..))| {
            // SAFETY: the only command buffer using these buffers has completed.
            unsafe {
                let read = KernelFixture::read_guarded(values, f32::NAN);
                assert!(read.iter().zip(inputs).all(|(a, b)| a.to_bits() == b.to_bits()), "values changed");
                (
                    KernelFixture::read_guarded(codes, CODE_SENTINEL),
                    KernelFixture::read_guarded(scales, SCALE_SENTINEL),
                    sums.as_ref().map_or(Vec::new(), |sums| KernelFixture::read_guarded(sums, SUM_SENTINEL)),
                )
            }
        })
        .collect()
}

/// Groups of `scale_group` values: each leader, which sets the group's largest magnitude, then its followers, as many as
/// fit after it, the rest zeros; a group of one holds each value alone. The row is padded with zeros to a multiple of
/// every group, a partial last tile included.
fn grouped_row(
    groups: &[(f32, Vec<f32>)],
    (scale_group, sum_group, _): (usize, Option<usize>, Int8CodeLayout),
) -> Vec<f32> {
    let mut row = Vec::new();
    for (leader, followers) in groups {
        if scale_group == 1 {
            row.push(*leader);
            row.extend(followers);
            continue;
        }
        for chunk in followers.chunks(scale_group - 1) {
            let group = [&[*leader][..], chunk].concat();
            row.extend([group.clone(), vec![0.0; scale_group - group.len()]].concat());
        }
    }
    let multiple = scale_group.max(sum_group.unwrap_or(1)).max(8);
    row.resize(row.len().div_ceil(multiple) * multiple + multiple, 0.0);
    row
}

/// The largest magnitude whose quotient by 127 rounds to `scale`.
fn maximum_for(scale: f32) -> f32 {
    let center = (scale * 127.0).to_bits();
    let candidates = (center - 64..=center + 64).rev().map(f32::from_bits);
    candidates.into_iter().find(|maximum| maximum / 127.0 == scale).expect("a maximum for the scale")
}

/// Boundary groups through the production quantizer for every model pair and smaller raw groups, exactly the canonical
/// CPU quantization: scales of exact powers of two (normal and subnormal) with quotients at midpoints k + 1/2 and their
/// neighbors (nextdown(1/2) included), infinite leaders giving scale 1 with the values themselves as quotients (midpoints,
/// saturation, the largest finite, subnormals), signed NaNs alone and next to finite values, signed zeros, tiny maxima
/// whose scale rounds to the smallest subnormal or underflows to 0 (nonzero codes ±127, zeros 0), the largest finite
/// maximum, and the three value and scale pairs where the native M2 quotient crossed a midpoint.
#[uzu_test]
fn quantizer_matches_canonical_at_boundaries() {
    let fixture = KernelFixture::new();
    let near = |value: f32| [f32::from_bits(value.to_bits() - 1), value, f32::from_bits(value.to_bits() + 1)];
    let signed = |values: Vec<f32>| values.iter().flat_map(|&value| [value, -value]).collect::<Vec<_>>();
    let midpoints = |unit: f32| {
        let midpoints = [0, 1, 2, 13, 52, 63, 125, 126].map(|k| (k as f32 + 0.5) * unit);
        signed(midpoints.into_iter().flat_map(near).collect())
    };
    let subnormal = f32::from_bits(1 << 9);
    let unit_extras = signed(vec![127.5, 300.0, f32::MAX, 1e-40, f32::from_bits(1), 0.0]);
    let nan = |bits: u32| f32::from_bits(bits);
    let recorded = [(0xc18d_115b, 0x3eab_f7e7), (0x422f_3898, 0x3f30_99cc), (0x407c_0349, 0x3e95_5748)];
    let mut groups = vec![
        (127.0 / 8.0, midpoints(1.0 / 8.0)),
        (127.0, midpoints(1.0)),
        (127.0 * subnormal, midpoints(subnormal)),
        (f32::INFINITY, [midpoints(1.0), unit_extras.clone()].concat()),
        (f32::NEG_INFINITY, unit_extras),
        (nan(0x7fc0_0000), vec![nan(0xffc0_0000), nan(0x7f80_0001), nan(0xffa5_a5a5)]),
        (nan(0xffc0_0000), vec![3.0, -1.5, 0.49999997, -0.0]),
        (0.0, vec![-0.0, 0.0, -0.0]),
        (f32::from_bits(63), signed(vec![f32::from_bits(5), f32::from_bits(1), 0.0])),
        (f32::from_bits(64), signed(vec![f32::from_bits(1), f32::from_bits(32), f32::from_bits(33)])),
        (f32::MAX, signed(vec![f32::MAX, f32::MAX / 254.0, 1.0])),
    ];
    for (value, scale) in recorded.map(|(value, scale)| (f32::from_bits(value), f32::from_bits(scale))) {
        groups.push((maximum_for(scale), signed(vec![value])));
    }
    let settings = direct_settings();
    let rows = settings.iter().map(|&setting| grouped_row(&groups, setting)).collect::<Vec<_>>();
    let cases = rows.iter().zip(&settings).map(|(row, &setting)| (&row[..], row.len(), setting)).collect::<Vec<_>>();
    let outputs = direct_outputs(&fixture, &cases);
    assert_eq!(outputs.len(), cases.len(), "one output per case");
    for (&(row, count, setting), gpu) in cases.iter().zip(&outputs) {
        assert_quantized(gpu, &quantized(row, count, setting), row, &format!("boundaries {}", label(setting)));
    }
    // The recorded cases at their canonical codes: 52.499996 rounds to code 52, 63.499996 to 63, while 13.49999959
    // rounds to 13.5 in FP32 first, so to 14.
    let (codes, scales, _) = quantized(&rows[0], rows[0].len(), settings[0]);
    for ((value, scale), code) in recorded.into_iter().zip([-52, 63, 14]) {
        let (value, scale) = (f32::from_bits(value), f32::from_bits(scale));
        let index = rows[0].iter().position(|&v| v.to_bits() == value.to_bits()).unwrap();
        assert_eq!(scales[index / settings[0].0].to_bits(), scale.to_bits(), "recorded scale");
        assert_eq!(i32::from(codes[settings[0].2.index(index % 8) + index - index % 8]), code, "recorded {value:e}");
    }
    fixture.assert_clean();
}

/// Random rows through the production quantizer for every direct setting, exactly the canonical CPU quantization:
/// mixed signs, significands and exponents over 2^-30 to 2^30. Prints how many CPU quotients fall within 8 ordered FP32
/// steps of a midpoint, where the shader takes its exact cold path.
#[uzu_test]
fn quantizer_matches_canonical_on_random_rows() {
    let fixture = KernelFixture::new();
    let (batch, count) = (256, 4096);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let values = (0..batch * count)
        .map(|_| {
            let bits = random();
            let exponent = (bits >> 32) % 61;
            f32::from_bits(((bits as u32) & 0x807f_ffff) | ((exponent as u32 + 97) << 23))
        })
        .collect::<Vec<_>>();
    let settings = direct_settings();
    let cases = settings.iter().map(|&setting| (&values[..], count, setting)).collect::<Vec<_>>();
    let mut window = 0;
    for (&(_, _, setting), gpu) in cases.iter().zip(direct_outputs(&fixture, &cases)) {
        let expected = quantized(&values, count, setting);
        assert_quantized(&gpu, &expected, &values, &format!("random {}", label(setting)));
        for (index, &value) in values.iter().enumerate() {
            let quotient = (value / expected.1[index / setting.0]).abs();
            let midpoint = quotient.floor() + 0.5;
            window +=
                usize::from(quotient < 127.5 && (quotient.to_bits() as i64 - midpoint.to_bits() as i64).abs() <= 8);
        }
    }
    eprintln!(
        "ActivationQuantization: {window} of {} quotients within the cold-path window",
        values.len() * cases.len()
    );
    fixture.assert_clean();
}

/// Construction rejects F16 and I32 for either type. `encode` rejects every optional argument missing where required or
/// present where not, also for empty rows, before recording anything; the same command buffer then completes valid work
/// and leaves the buffers of the rejected calls untouched.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    for (t, bias_t) in [
        (DataType::F16, DataType::F32),
        (DataType::I32, DataType::F32),
        (DataType::F32, DataType::F16),
        (DataType::BF16, DataType::I32),
    ] {
        let kernel = ActivationTransformVulkanKernel::new(
            &fixture.context,
            t,
            bias_t,
            ActivationTransformOp::InputRht,
            false,
            false,
            32,
            32,
            false,
        );
        assert!(
            matches!(
                kernel,
                Err(Error::KernelVariant {
                    kernel: "ActivationTransform",
                    ..
                })
            ),
            "{t:?}/{bias_t:?}"
        );
    }
    let quantization = ActivationQuantization::new(128, 64, true, Int8CodeLayout::GroupedByNibble).map(raw);
    let without_sums = ActivationQuantization::new(64, 64, false, Int8CodeLayout::Sequential).map(raw);
    let kernels = [
        (ActivationTransformOp::InputRht, false, false, None),
        (ActivationTransformOp::OutputRht, true, true, None),
        (ActivationTransformOp::InputRht, false, true, quantization),
        (ActivationTransformOp::InputRht, false, false, without_sums),
    ];
    let untouched = fixture.buffer(&[0u8; 4096]);
    let mut encoding = fixture.encoding();
    for (op, in_place, has_bias, quantization) in kernels {
        let kernel = vulkan_kernel::<f32, f32>(&fixture, op, in_place, has_bias, quantization);
        let quantized = quantization.is_some();
        let sums = quantization.is_some_and(|(_, sum_group, _)| sum_group.is_some());
        let present = [!in_place, !quantized, has_bias, quantized, quantized, sums];
        for (flip, count) in (0..present.len()).flat_map(|flip| [0, 128].map(|count| (flip, count))) {
            let mut arguments = present.map(|present| present.then_some((&untouched, 0..512)));
            arguments[flip] = arguments[flip].is_none().then_some((&untouched, 0..512));
            let [input, fp, bias, codes, scales, sums] = arguments;
            let encode = AssertUnwindSafe(|| unsafe {
                // SAFETY: never dispatched: the optional-argument assertion fails before recording.
                kernel.encode(input, fp, bias, codes, scales, sums, (&untouched, 0..512), 1, count, &mut encoding);
            });
            assert!(catch_unwind(encode).is_err(), "{op:?} in_place {in_place} argument {flip} count {count} accepted");
        }
    }
    let (input, factors) = (values::<f32>(64, 5), signs(64, 5));
    let (input_buffer, output, factor_buffer) =
        (fixture.buffer(&input), fixture.buffer(&[0.0f32; 64]), fixture.buffer(&factors));
    let kernel = vulkan_kernel::<f32, f32>(&fixture, ActivationTransformOp::InputRht, false, false, None);
    // SAFETY: the ranges hold 64 floats, 64 floats and 64 factors and do not alias.
    unsafe {
        kernel.encode(
            Some((&input_buffer, 0..256)),
            Some((&output, 0..256)),
            None,
            None,
            None,
            None,
            (&factor_buffer, 0..256),
            1,
            64,
            &mut encoding,
        );
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    let (output, untouched) = unsafe { (KernelFixture::read::<f32>(&output), KernelFixture::read::<u8>(&untouched)) };
    let case = (ActivationTransformOp::InputRht, false, None::<&[f32]>, None, &input[..], &factors[..], 1);
    check_bounds(
        &full_precision_bounds::<f32, f32>(&input, None, &factors, true),
        &cpu_outputs(case).0,
        &output,
        "valid work",
    );
    assert!(untouched.iter().all(|&byte| byte == 0), "a rejected call wrote");
    fixture.assert_clean();
}

/// The binding's preconditions. Construction rejects quantizing in place and quantization groups that are not powers
/// of two up to 128 (0, 3, 256; sum groups only with sums, but never 0), before creating anything, and admits smaller
/// raw powers of two. `encode` rejects rows not a multiple of 32 and of the quantization groups, also for empty batches,
/// before recording anything; the same command buffer then completes valid work, and rejected outputs stay untouched.
#[uzu_test]
fn rejects_violated_preconditions() {
    let fixture = KernelFixture::new();
    let (input_rht, quantize, with_sums) = (
        ActivationTransformOp::InputRht,
        ActivationTransformOp::Quantize,
        ActivationTransformOp::QuantizeWithGroupSums,
    );
    let new = |ops, in_place, scale_group, sum_group| {
        ActivationTransformVulkanKernel::new(
            &fixture.context,
            DataType::F32,
            DataType::F32,
            ops,
            false,
            in_place,
            scale_group,
            sum_group,
            false,
        )
    };
    let invalid = [
        (quantize, true, 32, 32),
        (with_sums, true, 32, 32),
        (quantize, false, 0, 32),
        (quantize, false, 3, 32),
        (quantize, false, 256, 32),
        (with_sums, false, 64, 0),
        (with_sums, false, 64, 3),
        (with_sums, false, 64, 256),
        (quantize, false, 64, 0),
    ];
    for (ops, in_place, scale_group, sum_group) in invalid {
        let result = new(ops, in_place, scale_group, sum_group);
        let rejected = matches!(
            result,
            Err(Error::KernelPrecondition {
                kernel: "ActivationTransform",
                ..
            })
        );
        assert!(rejected, "{ops:?} in_place {in_place} groups {scale_group}/{sum_group} accepted");
    }
    for (ops, in_place, scale_group, sum_group) in
        [(input_rht, true, 0, 0), (quantize, false, 1, 3), (with_sums, false, 2, 128)]
    {
        new(ops, in_place, scale_group, sum_group).expect("an admitted raw setting");
    }
    let untouched = fixture.buffer(&[0u8; 4096]);
    let mut encoding = fixture.encoding();
    let kernels =
        [(input_rht, 32, 32), (quantize, 64, 64), (with_sums, 32, 64)].map(|(ops, scale_group, sum_group)| {
            (ops, new(ops, false, scale_group, sum_group).expect("ActivationTransform"))
        });
    for ((ops, kernel), (batch, count)) in kernels.iter().zip([(1, 48), (1, 96), (0, 96)]) {
        let quantized = *ops != input_rht;
        let encode = AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the precondition fails before recording.
            kernel.encode(
                Some((&untouched, 0..512)),
                (!quantized).then_some((&untouched, 0..512)),
                None,
                quantized.then_some((&untouched, 0..512)),
                quantized.then_some((&untouched, 0..512)),
                (*ops == with_sums).then_some((&untouched, 0..512)),
                (&untouched, 0..512),
                batch,
                count,
                &mut encoding,
            );
        });
        assert!(catch_unwind(encode).is_err(), "{ops:?} {batch}x{count} accepted");
    }
    let (input, factors) = (values::<f32>(64, 8), signs(64, 8));
    let (input_buffer, output, factor_buffer) =
        (fixture.buffer(&input), fixture.buffer(&[0.0f32; 64]), fixture.buffer(&factors));
    // SAFETY: the ranges hold 64 floats, 64 floats and 64 factors and do not alias.
    unsafe {
        kernels[0].1.encode(
            Some((&input_buffer, 0..256)),
            Some((&output, 0..256)),
            None,
            None,
            None,
            None,
            (&factor_buffer, 0..256),
            1,
            64,
            &mut encoding,
        );
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    let (output, untouched) = unsafe { (KernelFixture::read::<f32>(&output), KernelFixture::read::<u8>(&untouched)) };
    let case = (input_rht, false, None::<&[f32]>, None, &input[..], &factors[..], 1);
    check_bounds(
        &full_precision_bounds::<f32, f32>(&input, None, &factors, true),
        &cpu_outputs(case).0,
        &output,
        "valid work",
    );
    assert!(untouched.iter().all(|&byte| byte == 0), "a rejected call wrote");
    fixture.assert_clean();
}

/// Run alone: `cargo test ... activation_transform_test::throughput -- --ignored --nocapture`. Construction cost, then
/// model-shaped rows for the full-precision input transform and both quantizations with the model's 128-element scale
/// groups, with fresh unchanging inputs.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float>(fixture: &KernelFixture) {
        let quantization = ActivationQuantization::new(128, 64, true, Int8CodeLayout::GroupedByNibble).map(raw);
        let without_sums = ActivationQuantization::new(128, 128, false, Int8CodeLayout::Sequential).map(raw);
        let settings = [
            (ActivationTransformOp::InputRht, None),
            (ActivationTransformOp::InputRht, without_sums),
            (ActivationTransformOp::InputRht, quantization),
        ];
        let mut construction = (0..11)
            .map(|_| {
                let start = Instant::now();
                vulkan_kernel::<T, T>(fixture, settings[2].0, false, false, settings[2].1);
                start.elapsed()
            })
            .collect::<Vec<_>>();
        let first = construction[0];
        construction.sort();
        eprintln!(
            "ActivationTransform {:?} construction: first {first:?}, median of 11 {:?}",
            T::data_type(),
            construction[5]
        );
        for rows in [1, 128, 1024] {
            for count in [4096, 14336] {
                let n = rows * count;
                let (input, factors) = (fixture.buffer(&values::<T>(n, 1)), fixture.buffer(&signs(count, 1)));
                let (fp, codes) = (fixture.buffer(&vec![T::zero(); n]), fixture.buffer(&vec![0i8; n]));
                let (scales, sums) = (fixture.buffer(&vec![0f32; n / 32]), fixture.buffer(&vec![0i32; n / 32]));
                for (op, quantization) in settings {
                    let kernel = vulkan_kernel::<T, T>(fixture, op, false, false, quantization);
                    let bytes = |size: usize, divisor: usize| 0..(n / divisor * size) as u64;
                    let with_sums = quantization.is_some_and(|(_, sum_group, _)| sum_group.is_some());
                    let (gpu, wall) = fixture.median_times(|encoding| unsafe {
                        // SAFETY: every range holds the elements of `rows` rows of `count`; outputs do not alias.
                        kernel.encode(
                            Some((&input, bytes(size_of::<T>(), 1))),
                            quantization.is_none().then(|| (&fp, bytes(size_of::<T>(), 1))),
                            None,
                            quantization.map(|_| (&codes, bytes(1, 1))),
                            quantization.map(|(scale_group, ..)| (&scales, bytes(4, scale_group))),
                            with_sums.then(|| (&sums, bytes(4, 64))),
                            (&factors, 0..(count * 4) as u64),
                            rows as u32,
                            count as u32,
                            encoding,
                        );
                    });
                    let label = quantization.map_or("full precision".to_string(), label);
                    eprintln!(
                        "MEASURE ActivationTransform {:?} {rows}x{count} {label}: median of 10 after 3 warm-up: GPU {gpu:?}, wall {wall:?}",
                        T::data_type()
                    );
                }
            }
        }
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
